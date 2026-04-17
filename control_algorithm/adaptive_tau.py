import math
import numpy as np
import torch
from util.utils import recv_msg, send_msg, moving_average


class ControlAlgAdaptiveTauServer:
    def __init__(self, is_adapt_local, client_sock_all, n_nodes, control_param_phi,
                 moving_average_holding_param):
        self.is_adapt_local = is_adapt_local
        self.client_sock_all = client_sock_all
        self.n_nodes = n_nodes
        self.control_param_phi = control_param_phi
        self.moving_average_holding_param = moving_average_holding_param

        self.beta_adapt_mvaverage = None
        self.delta_adapt_mvaverage = None
        self.omega_adapt_mvaverage = None
        self.rho_adapt_mvaverage = None
        self.tau_per_device = {}

        # 新增：per-client收敛界参数存储（对齐附录D-F）
        self.per_client_conv_params = {
            "Ld": [0.0] * n_nodes,  # 客户端d的Lipschitz常数Ld
            "rho_cd": [0.0] * n_nodes,  # 客户端d的损失ρ-Lipschitz常数
            "omega_cd": [0.0] * n_nodes,  # 客户端d的ω_cd=1/||θ_cd - θ*||²
            "s_d": [0.0] * n_nodes  # 客户端d的数据量占比s_d
        }

    def compute_tau_cd(self, t, alpha=0.1, beta=0.9, Ld=0.0):
        """计算附录D Lemma1的τ_[c]^d(t)（客户端专属偏差系数）"""
        # 附录公式参数：W=1+3β+αLd, Z=1+β+αLd, X=√(Z²+4β), Y=2β+αLd
        W = 1 + 3 * beta + alpha * Ld
        Z = 1 + beta + alpha * Ld
        X = np.sqrt(np.square(Z) + 4 * beta)  # 修正原代码X计算错误（附录中X=√[(1+β+αLd)²+4β]）
        Y = 2 * beta + alpha * Ld

        # 附录核心公式：τ_cd(t) = (W+X)/(2XY)*(X+Z)/2^t - (W-X)/(2XY)*(Z-X)/2^t - 1/Y
        base1 = (X + Z) / 2
        base2 = (Z - X) / 2
        term1 = (W + X) / (2 * X * Y) * np.power(base1, t)
        term2 = (W - X) / (2 * X * Y) * np.power(base2, t)
        term3 = 1 / Y
        tau_cd_t = term1 - term2 - term3
        return max(tau_cd_t, 0.0)  # 避免数值误差导致的负值

    def compute_epsilon0(self, tau_candidate, alpha=0.1, beta=0.9, client_idx=None):
        """计算附录F Theorem3的ε0（收敛界上界核心项）

        参数:
            tau_candidate: 当前候选迭代次数λ_cd
            alpha: 学习率α
            beta: 动量系数β
            client_idx: 可选，特定客户端索引，None表示计算全局收敛界

        返回:
            total_bound: 收敛界上界
        """
        sum_slambda_omega_zeta = 0.0  # ∑s_d * λ_cd * ω_cd * ζ_cd
        sum_s_delta_rho_tau = 0.0  # ∑s_d * δ_cd * ρ_cd * τ_cd(λ_cd)

        # 确定要计算的客户端列表
        client_list = [client_idx] if client_idx is not None else range(self.n_nodes)

        for d in client_list:
            s_d = self.per_client_conv_params["s_d"][d]
            Ld = self.per_client_conv_params["Ld"][d]
            rho_cd = self.per_client_conv_params["rho_cd"][d]
            omega_cd = self.per_client_conv_params["omega_cd"][d]
            delta_cd = self.delta_adapt_local[d]  # 客户端d的梯度分歧δ_cd
            lambda_cd = tau_candidate  # 当前候选迭代次数λ_cd

            # 计算ζ_cd=α(1 - αLd/2)（附录E Lemma2）
            zeta_cd = alpha * (1 - (alpha * Ld) / 2)
            # 累加∑s_dλ_cdω_cdζ_cd
            sum_slambda_omega_zeta += s_d * lambda_cd * omega_cd * zeta_cd
            # 计算τ_cd(λ_cd)并累加∑s_dδ_cdρ_cdτ_cd
            tau_cd = self.compute_tau_cd(lambda_cd, alpha, beta, Ld)
            sum_s_delta_rho_tau += s_d * delta_cd * rho_cd * tau_cd

        # 附录F Eqn.49：ε0 = [1 + √(1+4α*∑sλωζ*∑sδρτ)] / (2∑sλωζ)
        numerator = 1 + np.sqrt(1 + 4 * alpha * sum_slambda_omega_zeta * sum_s_delta_rho_tau)
        denominator = 2 * sum_slambda_omega_zeta + 1e-10  # 避免除零
        epsilon0 = numerator / denominator

        # 总收敛界上界：ε0 + α*∑sδρτ（附录F Theorem3 Eqn.52）
        total_bound = epsilon0 + alpha * sum_s_delta_rho_tau
        return total_bound

    def compute_new_tau(self, data_size_local_all, data_size_total, tau, delt_f, gam_f):
        torch.cuda.empty_cache()
        beta_adapt = 0.0
        delta_adapt = 0.0
        omega_adapt = 0.0
        rho_adapt = 0.0
        global_grad_global_weight = None
        self.beta_adapt_local = []
        self.rho_adapt_local = []
        self.omega_adapt_local = []
        self.delta_adapt_local = []
        a = 0.1  # 学习率α（对齐客户端HeavyBallMGD）
        b = 0.9  # 动量系数β（对齐客户端HeavyBallMGD）
        tau_new = []

        local_grad_global_weight_all = []
        control_param_computed = False

        # 初始化客户端数据量占比s_d（对齐附录s_d定义）
        for d in range(self.n_nodes):
            self.per_client_conv_params["s_d"][d] = data_size_local_all[d] / data_size_total

        # 第一步：遍历所有客户端，接收梯度并累加收敛界参数
        for n in range(self.n_nodes):
            # 接收客户端计算状态
            msg = recv_msg(self.client_sock_all[n], 'MSG_CONTROL_PARAM_COMPUTED_CLIENT_TO_SERVER')
            control_param_computed_this_client = msg[1]
            control_param_computed = control_param_computed or control_param_computed_this_client

            if control_param_computed_this_client:
                # 接收客户端的beta/rho/梯度/omega（新增Ld、rho_cd映射）
                msg = recv_msg(self.client_sock_all[n], 'MSG_BETA_RHO_GRAD_CLIENT_TO_SERVER')
                beta_local = msg[1]
                rho_local = msg[2]
                local_grad_global_weight = msg[3]
                omega_local = msg[4]

                self.beta_adapt_local.append(beta_local)
                self.rho_adapt_local.append(rho_local)
                self.omega_adapt_local.append(omega_local)
                local_grad_global_weight_all.append(local_grad_global_weight)

                # 映射客户端参数到收敛界参数（无额外上传时用现有参数近似）
                self.per_client_conv_params["Ld"][n] = beta_local  # Ld≈beta_adapt（梯度平滑系数近似）
                self.per_client_conv_params["rho_cd"][n] = rho_local  # rho_cd=rho_adapt（损失Lipschitz）
                self.per_client_conv_params["omega_cd"][n] = omega_local  # omega_cd=omega_adapt

                # 初始化全局梯度结构（仅第一次）
                from config import adapt_layers
                if global_grad_global_weight is None:
                    global_grad_global_weight = {
                        "img": {level: [] for level in adapt_layers},
                        "img_kappa_hypernet": {
                            "global_encoder": [],
                            "level_predictors": {level: [] for level in adapt_layers},
                            "level_embeddings": {level: [] for level in adapt_layers}
                        }
                    }
                    # 初始化梯度零张量
                    for adapt_type in ["img", "img_kappa_hypernet"]:
                        if adapt_type == "img":
                            for level in adapt_layers:
                                if level in local_grad_global_weight.get(adapt_type, {}):
                                    for grad_tensor in local_grad_global_weight[adapt_type][level]:
                                        global_grad_global_weight[adapt_type][level].append(
                                            torch.zeros_like(grad_tensor)
                                        )
                        else:
                            hypernet_local = local_grad_global_weight.get(adapt_type, {})
                            if "global_encoder" in hypernet_local:
                                for grad_tensor in hypernet_local["global_encoder"]:
                                    global_grad_global_weight[adapt_type]["global_encoder"].append(
                                        torch.zeros_like(grad_tensor)
                                    )
                            if "level_predictors" in hypernet_local:
                                for level in adapt_layers:
                                    if level in hypernet_local["level_predictors"]:
                                        for grad_tensor in hypernet_local["level_predictors"][level]:
                                            global_grad_global_weight[adapt_type]["level_predictors"][level].append(
                                                torch.zeros_like(grad_tensor)
                                            )
                            if "level_embeddings" in hypernet_local:
                                for level in adapt_layers:
                                    if level in hypernet_local["level_embeddings"]:
                                        for grad_tensor in hypernet_local["level_embeddings"][level]:
                                            global_grad_global_weight[adapt_type]["level_embeddings"][level].append(
                                                torch.zeros_like(grad_tensor)
                                            )

                # 加权累加梯度（按数据量）
                for adapt_type in ["img", "img_kappa_hypernet"]:
                    if adapt_type == "img":
                        for level in adapt_layers:
                            if (level in local_grad_global_weight.get(adapt_type, {}) and
                                    level in global_grad_global_weight[adapt_type]):
                                for i, grad_tensor in enumerate(local_grad_global_weight[adapt_type][level]):
                                    weighted_grad = data_size_local_all[n] * grad_tensor
                                    global_grad_global_weight[adapt_type][level][i] += weighted_grad
                    else:
                        hypernet_local = local_grad_global_weight.get(adapt_type, {})
                        # Global encoder
                        if "global_encoder" in hypernet_local and "global_encoder" in global_grad_global_weight[
                            adapt_type]:
                            for i, grad_tensor in enumerate(hypernet_local["global_encoder"]):
                                weighted_grad = data_size_local_all[n] * grad_tensor
                                global_grad_global_weight[adapt_type]["global_encoder"][i] += weighted_grad
                        # Level predictors
                        if "level_predictors" in hypernet_local and "level_predictors" in global_grad_global_weight[
                            adapt_type]:
                            for level in adapt_layers:
                                if level in hypernet_local["level_predictors"] and level in \
                                        global_grad_global_weight[adapt_type]["level_predictors"]:
                                    for i, grad_tensor in enumerate(hypernet_local["level_predictors"][level]):
                                        weighted_grad = data_size_local_all[n] * grad_tensor
                                        global_grad_global_weight[adapt_type]["level_predictors"][level][
                                            i] += weighted_grad
                        # Level embeddings
                        if "level_embeddings" in hypernet_local and "level_embeddings" in global_grad_global_weight[
                            adapt_type]:
                            for level in adapt_layers:
                                if level in hypernet_local["level_embeddings"] and level in \
                                        global_grad_global_weight[adapt_type]["level_embeddings"]:
                                    for i, grad_tensor in enumerate(hypernet_local["level_embeddings"][level]):
                                        weighted_grad = data_size_local_all[n] * grad_tensor
                                        global_grad_global_weight[adapt_type]["level_embeddings"][level][
                                            i] += weighted_grad

                # 累加beta/rho/omega（按数据量加权）
                beta_adapt += data_size_local_all[n] * beta_local
                rho_adapt += data_size_local_all[n] * rho_local
                omega_adapt += data_size_local_all[n] * omega_local

        # 第二步：所有客户端处理完成后，归一化全局梯度和收敛界参数
        if global_grad_global_weight is not None and control_param_computed:
            # 归一化全局梯度
            for adapt_type in ["img", "img_kappa_hypernet"]:
                if adapt_type == "img":
                    for level in adapt_layers:
                        for i in range(len(global_grad_global_weight[adapt_type][level])):
                            global_grad_global_weight[adapt_type][level][i] /= data_size_total
                else:
                    for i in range(len(global_grad_global_weight[adapt_type]["global_encoder"])):
                        global_grad_global_weight[adapt_type]["global_encoder"][i] /= data_size_total
                    for level in adapt_layers:
                        for i in range(len(global_grad_global_weight[adapt_type]["level_predictors"][level])):
                            global_grad_global_weight[adapt_type]["level_predictors"][level][i] /= data_size_total
                        for i in range(len(global_grad_global_weight[adapt_type]["level_embeddings"][level])):
                            global_grad_global_weight[adapt_type]["level_embeddings"][level][i] /= data_size_total

            # 归一化beta/rho/omega（移动平均前）
            beta_adapt /= data_size_total
            rho_adapt /= data_size_total
            omega_adapt /= data_size_total

            # 计算每个客户端的delta_adapt_local（梯度分歧δ_cd）
            for i in range(self.n_nodes):
                delta_val = 0.0
                for adapt_type in ["img", "img_kappa_hypernet"]:
                    if adapt_type == "img":
                        for level in adapt_layers:
                            if (adapt_type in local_grad_global_weight_all[i] and
                                    level in local_grad_global_weight_all[i][adapt_type] and
                                    adapt_type in global_grad_global_weight and
                                    level in global_grad_global_weight[adapt_type]):
                                for j in range(len(local_grad_global_weight_all[i][adapt_type][level])):
                                    local_grad = local_grad_global_weight_all[i][adapt_type][level][j]
                                    global_grad = global_grad_global_weight[adapt_type][level][j]
                                    if isinstance(local_grad, torch.Tensor) and isinstance(global_grad, torch.Tensor):
                                        local_grad = local_grad.to(global_grad.device)
                                        diff = local_grad - global_grad
                                        delta_val += torch.sum(diff ** 2).item()
                    else:
                        if adapt_type in local_grad_global_weight_all[i] and adapt_type in global_grad_global_weight:
                            local_hyper_grad = local_grad_global_weight_all[i][adapt_type]
                            global_hyper_grad = global_grad_global_weight[adapt_type]
                            # Global encoder
                            if "global_encoder" in local_hyper_grad and "global_encoder" in global_hyper_grad:
                                for j in range(len(local_hyper_grad["global_encoder"])):
                                    local_grad = local_hyper_grad["global_encoder"][j]
                                    global_grad = global_hyper_grad["global_encoder"][j]
                                    if isinstance(local_grad, torch.Tensor) and isinstance(global_grad, torch.Tensor):
                                        local_grad = local_grad.to(global_grad.device)
                                        diff = local_grad - global_grad
                                        delta_val += torch.sum(diff ** 2).item()
                            # Level predictors
                            if "level_predictors" in local_hyper_grad and "level_predictors" in global_hyper_grad:
                                for level in adapt_layers:
                                    if level in local_hyper_grad["level_predictors"] and level in global_hyper_grad[
                                        "level_predictors"]:
                                        for j in range(len(local_hyper_grad["level_predictors"][level])):
                                            local_grad = local_hyper_grad["level_predictors"][level][j]
                                            global_grad = global_hyper_grad["level_predictors"][level][j]
                                            if isinstance(local_grad, torch.Tensor) and isinstance(global_grad,
                                                                                                   torch.Tensor):
                                                local_grad = local_grad.to(global_grad.device)
                                                diff = local_grad - global_grad
                                                delta_val += torch.sum(diff ** 2).item()
                            # Level embeddings
                            if "level_embeddings" in local_hyper_grad and "level_embeddings" in global_hyper_grad:
                                for level in adapt_layers:
                                    if level in local_hyper_grad["level_embeddings"] and level in global_hyper_grad[
                                        "level_embeddings"]:
                                        for j in range(len(local_hyper_grad["level_embeddings"][level])):
                                            local_grad = local_hyper_grad["level_embeddings"][level][j]
                                            global_grad = global_hyper_grad["level_embeddings"][level][j]
                                            if isinstance(local_grad, torch.Tensor) and isinstance(global_grad,
                                                                                                   torch.Tensor):
                                                local_grad = local_grad.to(global_grad.device)
                                                diff = local_grad - global_grad
                                                delta_val += torch.sum(diff ** 2).item()
                delta_val = math.sqrt(max(delta_val, 1e-15))  # 数值稳定性保护
                self.delta_adapt_local.append(delta_val)
                delta_adapt += data_size_local_all[i] * delta_val

            # 归一化delta_adapt
            delta_adapt /= data_size_total

            # 移动平均更新（处理初始值None）
            self.beta_adapt_mvaverage = moving_average(self.beta_adapt_mvaverage, beta_adapt,
                                                       self.moving_average_holding_param) if self.beta_adapt_mvaverage is not None else beta_adapt
            self.delta_adapt_mvaverage = moving_average(self.delta_adapt_mvaverage, delta_adapt,
                                                        self.moving_average_holding_param) if self.delta_adapt_mvaverage is not None else delta_adapt
            self.rho_adapt_mvaverage = moving_average(self.rho_adapt_mvaverage, rho_adapt,
                                                      self.moving_average_holding_param) if self.rho_adapt_mvaverage is not None else rho_adapt
            self.omega_adapt_mvaverage = moving_average(self.omega_adapt_mvaverage, omega_adapt,
                                                        self.moving_average_holding_param) if self.omega_adapt_mvaverage is not None else omega_adapt

            print('betaAdapt_mvaverage =', self.beta_adapt_mvaverage)
            print('deltaAdapt_mvaverage =', self.delta_adapt_mvaverage)
            print('rhoAdapt_mvaverage =', self.rho_adapt_mvaverage)
            print('omegaAdapt_mvaverage =', self.omega_adapt_mvaverage)

            # 计算新tau：基于论文收敛界优化（限制∈[2,8]）
            if self.is_adapt_local:
                # 为每个客户端独立搜索最优tau
                for n in range(self.n_nodes):
                    min_converge_bound = float('inf')
                    best_tau_local = 2  # 初始最优tau

                    # 线性搜索[2,8]内最小化该客户端收敛界的最优tau
                    for tau_candidate in range(2, 9):
                        # 计算当前候选tau对该客户端的收敛界上界
                        current_bound = self.compute_epsilon0(tau_candidate, alpha=a, beta=b, client_idx=n)
                        if current_bound < min_converge_bound:
                            min_converge_bound = current_bound
                            best_tau_local = tau_candidate
                    tau_new.append(best_tau_local)

                    # 记录每个设备的tau
                    self.tau_per_device[n] = best_tau_local

                print(f"基于收敛界优化的客户端独立最优tau：{tau_new}")
            else:
                # 非自适应模式：固定tau∈[2,8]
                tau_new = [max(2, min(8, tau)) for _ in range(self.n_nodes)]
        else:
            # control_param_computed为False时，默认tau∈[2,8]
            tau_new = [max(2, min(8, tau)) for _ in range(self.n_nodes)]

        # 最终强制限制tau∈[2,8]（双重保障）
        tau_new = [max(2, min(8, t)) for t in tau_new]

        return tau_new, delt_f, gam_f

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['client_sock_all']
        return state


class ControlAlgAdaptiveTauClient:
    def __init__(self):
        self.w_last_local_last_round = None
        self.grad_last_local_last_round = None
        self.loss_last_local_last_round = None

    def init_new_round(self):
        self.control_param_computed = False
        self.beta_adapt = None
        self.rho_adapt = None
        self.grad_last_global = None
        self.omega_adapt = None

    def update_after_each_local(self, tau, grad_global):
        self.grad_last_global = grad_global
        return False

    def update_after_all_local(self, last_grad, last_loss,
                               w, w_last_global, loss_last_global, w_global_min_loss, omega):
        if w_global_min_loss is None:
            w_global_min_loss = w_last_global

        # 提取参数
        w_last_global_params = [param.detach().clone() for param in w_last_global.parameters()]
        w_global_min_loss_params = [param.detach().clone() for param in w_global_min_loss.parameters()]
        w_params = [param.detach().clone() for param in w.parameters()]

        # 计算beta/rho/omega（对齐服务器收敛界参数需求）
        if (self.w_last_local_last_round is not None and self.grad_last_local_last_round is not None and
                self.loss_last_local_last_round is not None):

            w_last_local_last_round_params = [param.detach().clone() for param in
                                              self.w_last_local_last_round.parameters()]

            # 初始化梯度差结构
            from config import adapt_layers
            grad_diff = {
                "img": {level: [] for level in adapt_layers},
                "img_kappa_hypernet": {
                    "global_encoder": [],
                    "level_predictors": {level: [] for level in adapt_layers},
                    "level_embeddings": {level: [] for level in adapt_layers}
                }
            }

            # 计算梯度差
            for adapt_type in ["img", "img_kappa_hypernet"]:
                if adapt_type == "img":
                    for level in adapt_layers:
                        if (adapt_type in last_grad and level in last_grad[adapt_type] and
                                adapt_type in self.grad_last_global and level in self.grad_last_global[adapt_type]):
                            for local_grad, global_grad in zip(last_grad[adapt_type][level],
                                                               self.grad_last_global[adapt_type][level]):
                                local_grad = local_grad.to(global_grad.device)
                                grad_diff[adapt_type][level].append(local_grad - global_grad)
                else:
                    if (adapt_type in last_grad) and (adapt_type in self.grad_last_global):
                        local_hyper_grad = last_grad[adapt_type]
                        global_hyper_grad = self.grad_last_global[adapt_type]
                        # Global encoder
                        for local_grad, global_grad in zip(local_hyper_grad["global_encoder"],
                                                           global_hyper_grad["global_encoder"]):
                            local_grad = local_grad.to(global_grad.device)
                            grad_diff[adapt_type]["global_encoder"].append(local_grad - global_grad)
                        # Level predictors
                        for level in adapt_layers:
                            for local_grad, global_grad in zip(local_hyper_grad["level_predictors"][level],
                                                               global_hyper_grad["level_predictors"][level]):
                                local_grad = local_grad.to(global_grad.device)
                                grad_diff[adapt_type]["level_predictors"][level].append(local_grad - global_grad)
                        # Level embeddings
                        for level in adapt_layers:
                            for local_grad, global_grad in zip(local_hyper_grad["level_embeddings"][level],
                                                               global_hyper_grad["level_embeddings"][level]):
                                local_grad = local_grad.to(global_grad.device)
                                grad_diff[adapt_type]["level_embeddings"][level].append(local_grad - global_grad)

            # 计算梯度范数（beta_adapt=梯度差/参数差，近似Ld）
            grad_norm = 0.0
            for adapt_type in ["img", "img_kappa_hypernet"]:
                if adapt_type == "img":
                    for level in adapt_layers:
                        for tensor in grad_diff[adapt_type][level]:
                            if isinstance(tensor, torch.Tensor):
                                grad_norm += torch.sum(tensor ** 2).item()
                else:
                    for tensor in grad_diff[adapt_type]["global_encoder"]:
                        if isinstance(tensor, torch.Tensor):
                            grad_norm += torch.sum(tensor ** 2).item()
                    for level in adapt_layers:
                        for tensor in grad_diff[adapt_type]["level_predictors"][level]:
                            if isinstance(tensor, torch.Tensor):
                                grad_norm += torch.sum(tensor ** 2).item()
                        for tensor in grad_diff[adapt_type]["level_embeddings"][level]:
                            if isinstance(tensor, torch.Tensor):
                                grad_norm += torch.sum(tensor ** 2).item()
            grad_norm = math.sqrt(max(grad_norm, 1e-15))

            # 计算参数差范数
            tmp_norm_sq = 0.0
            for p_local, p_global in zip(w_last_local_last_round_params, w_last_global_params):
                # 确保所有张量在同一设备上
                device = p_local.device
                p_global = p_global.to(device)
                diff = p_local - p_global
                tmp_norm_sq += torch.sum(diff ** 2).item()
            tmp_norm = math.sqrt(max(tmp_norm_sq, 1e-15))

            # 计算beta_adapt（近似Ld，对齐服务器收敛界Ld需求）
            self.beta_adapt = grad_norm / tmp_norm if tmp_norm > 1e-15 else 1e5

            # 计算rho_adapt（近似rho_cd，对齐服务器收敛界rho_cd需求）
            loss_diff = self.loss_last_local_last_round - loss_last_global
            self.rho_adapt = np.linalg.norm(loss_diff) / tmp_norm if tmp_norm > 1e-15 else 0.0
            self.rho_adapt = 0.0 if np.isnan(self.rho_adapt) or np.isinf(self.rho_adapt) else self.rho_adapt

            # 计算omega_adapt（近似omega_cd=1/||θ_cd - θ*||²，对齐服务器需求）
            total_sq_diff = 0.0
            for p1, p2 in zip(w_params, w_global_min_loss_params):
                # 确保所有张量在同一设备上
                device = p1.device
                p2 = p2.to(device)
                diff = p1 - p2
                total_sq_diff += torch.sum(diff ** 2).item()
            norm_sq = max(total_sq_diff, 1e-15)
            self.omega_adapt = 1.0 / norm_sq
            self.omega_adapt = max(self.omega_adapt, omega)

            print('betaAdapt =', self.beta_adapt)
            self.control_param_computed = True

        # 更新状态
        self.grad_last_local_last_round = last_grad
        self.loss_last_local_last_round = last_loss
        self.w_last_local_last_round = w

    def send_to_server(self, sock):
        # 发送计算状态
        msg = ['MSG_CONTROL_PARAM_COMPUTED_CLIENT_TO_SERVER', self.control_param_computed]
        send_msg(sock, msg)

        # 发送beta/rho/梯度/omega（供服务器映射为收敛界参数）
        if self.control_param_computed:
            msg = ['MSG_BETA_RHO_GRAD_CLIENT_TO_SERVER',
                   self.beta_adapt,
                   self.rho_adapt,
                   self.grad_last_global,
                   self.omega_adapt]
            send_msg(sock, msg)