import numpy as np
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from control_algorithm.adaptive_tau import ControlAlgAdaptiveTauServer
from config import *

class CollectStatistics:
    def __init__(self, results_file_name=os.path.dirname(__file__)+'/results.csv', is_single_run=False):
        self.results_file_name = results_file_name
        self.is_single_run = is_single_run

        if not os.path.exists(os.path.dirname(results_file_name)):
            os.makedirs(os.path.dirname(results_file_name))
        if is_single_run:
            with open(results_file_name, 'a') as f:
                f.write(
                    'tValue,lossValue,betaAdapt,deltaAdapt,rhoAdapt,tau\n')
                f.close()
        else:
            with open(results_file_name, 'a') as f:
                f.write(
                    'tau_setup,avg_tau,stddev_tau,' +
                    'avg_betaAdapt,stddev_betaAdapt,' +
                    'avg_deltaAdapt,stddev_deltaAdapt,avg_rhoAdapt,stddev_rhoAdapt,' +
                    'total_time_recomputed,E_total\n')
                f.close()

    def init_stat_new_global_round(self):

        if self.is_single_run:
            self.loss_values = []
            self.prediction_accuracies = []
            self.t_values = []

        self.taus = []
        self.each_locals = []
        self.each_globals = []
        self.beta_adapts = []
        self.delta_adapts = []
        self.rho_adapts = []

    def init_stat_new_global_round_comp(self):

        if self.is_single_run:
            self.loss_values = []
            self.prediction_accuracies = []
            self.t_values = []
            self.k = []
            self.immediate_cost =[]

    def collect_stat_end_local_round(self, tau, control_alg,
                                     total_time_recomputed, loss_last_global):

        self.taus.append(tau)  # 记录使用的 tau

        # 记录控制算法参数
        if control_alg is not None:
            if isinstance(control_alg, ControlAlgAdaptiveTauServer):
                if control_alg.beta_adapt_mvaverage is not None:
                    self.beta_adapts.append(control_alg.beta_adapt_mvaverage)
                elif self.is_single_run:
                    self.beta_adapts.append(np.nan)

                if control_alg.delta_adapt_mvaverage is not None:
                    self.delta_adapts.append(control_alg.delta_adapt_mvaverage)
                elif self.is_single_run:
                    self.delta_adapts.append(np.nan)

                if control_alg.rho_adapt_mvaverage is not None:
                    self.rho_adapts.append(control_alg.rho_adapt_mvaverage)
                elif self.is_single_run:
                    self.rho_adapts.append(np.nan)
        else:
            if self.is_single_run:
                self.beta_adapts.append(np.nan)
                self.delta_adapts.append(np.nan)
                self.rho_adapts.append(np.nan)

        if self.is_single_run:
            loss_value = loss_last_global

            print(f"***** Validation completed at time: {total_time_recomputed}")

            # 保存结果
            with open(self.results_file_name, 'a') as f:
                f.write(f"{total_time_recomputed},{loss_value},"
                        f"{self.beta_adapts[-1]},{self.delta_adapts[-1]},{self.rho_adapts[-1]},"
                        f"{tau}\n")

    def collect_stat_end_local_round_comp(self, case, num_iter, model, train_image, train_label, test_image, test_label,
                                          w_global, total_time_recomputed, k=None, cost=None):
        if self.is_single_run:
            loss_value = model.loss(train_image, train_label, w_global)
            self.loss_values.append(loss_value)

            prediction_accuracy = model.accuracy(test_image, test_label, w_global)
            self.prediction_accuracies.append(prediction_accuracy)

            self.t_values.append(total_time_recomputed)
            self.k.append(k)
            self.immediate_cost.append(cost)

            print("***** lossValue: " + str(loss_value))

            with open(self.results_file_name, 'a') as f:
                f.write(str(case) + ',' + str(num_iter) + ',' + str(total_time_recomputed) + ',' + str(loss_value) + ','
                        + str(prediction_accuracy) + ',' + str(k) + ',' + str(cost) + '\n')
                f.close()

    def collect_stat_end_global_round(self, tau_setup, total_time, total_time_recomputed, E_total):

        generated_images_dir = os.path.join(args.output_dir, f"generated_{tau_setup}")
        os.makedirs(generated_images_dir, exist_ok=True)

        if not self.is_single_run:
            taus_array = np.array(self.taus)
            avg_tau = np.mean(taus_array)
            stddev_tau = np.std(taus_array)
            avg_beta_adapt = np.mean(np.array(self.beta_adapts))
            stddev_beta_adapt = np.std(np.array(self.beta_adapts))
            avg_delta_adapt = np.mean(np.array(self.delta_adapts))
            stddev_delta_adapt = np.std(np.array(self.delta_adapts))
            avg_rho_adapt = np.mean(np.array(self.rho_adapts))
            stddev_rho_adapt = np.std(np.array(self.rho_adapts))

            with open(self.results_file_name, 'a') as f:
                f.write(f"{tau_setup},"
                        f"{avg_tau},{stddev_tau},"
                        f"{avg_beta_adapt},{stddev_beta_adapt},"
                        f"{avg_delta_adapt},{stddev_delta_adapt},"
                        f"{avg_rho_adapt},{stddev_rho_adapt},"
                        f"{total_time_recomputed},{E_total}\n")

        print(f'total time: {total_time}')