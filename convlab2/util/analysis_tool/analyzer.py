import os
from convlab2.dialog_agent import BiSession, PipelineAgent
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from pprint import pprint
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from convlab2.util.analysis_tool.helper import Reporter
from tqdm import tqdm, trange
import logging


class Analyzer:
    def __init__(self, user_agent, dataset='multiwoz'):
        self.user_agent = user_agent
        self.dataset = dataset

    def build_sess(self, sys_agent):
        if self.dataset == 'multiwoz':
            evaluator = MultiWozEvaluator()
        else:
            evaluator = None

        if evaluator is None:
            self.sess = None
        else:
            self.sess = BiSession(sys_agent=sys_agent, user_agent=self.user_agent, kb_query=None, evaluator=evaluator)
        return self.sess

    def sample_dialog(self, sys_agent):
        sess = self.build_sess(sys_agent)
        sys_response = '' if self.user_agent.nlu else []
        sess.init_session()
        print('init goal:')
        pprint(sess.evaluator.goal)
        print('-'*50)
        for i in range(40):
            sys_response, user_response, session_over, reward = sess.next_turn(sys_response)
            print('user:', user_response)
            # print('user in da:', sess.user_agent.get_in_da())
            # print('user out da:', sess.user_agent.get_out_da())
            print('sys:', sys_response)
            # print('sys in da:', sess.sys_agent.get_in_da())
            # print('sys out da:', sess.sys_agent.get_out_da())
            print()
            if session_over is True:
                break
        print('task complete:', sess.user_agent.policy.policy.goal.task_complete())
        print('task success:', sess.evaluator.task_success())
        print('book rate:', sess.evaluator.book_rate())
        print('inform precision/recall/f1:', sess.evaluator.inform_F1())
        print(f"percentage of domains that satisfies the database constraints: {sess.evaluator.final_goal_analyze()}")
        print('-' * 50)
        print('final goal:')
        pprint(sess.evaluator.goal)
        print('=' * 100)

    def comprehensive_analyze(self, sys_agent, model_name, total_dialog=100):
        sess = self.build_sess(sys_agent)

        goal_seeds = [random.randint(1,100000) for _ in range(total_dialog)]
        precision = []
        recall = []
        f1 = []
        match = []
        suc_num = 0
        complete_num = 0
        turn_num = 0
        turn_suc_num = 0
        num_domains = 0
        num_domains_satisfying_constraints = 0
        num_dialogs_satisfying_constraints = 0

        reporter = Reporter(model_name)
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        if not os.path.exists('results'):
            os.mkdir('results')
        output_dir = os.path.join('results', model_name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        f = open(os.path.join(output_dir, 'res.txt'), 'w')
        flog = open(os.path.join(output_dir, 'log.txt'), 'w')

        for j in tqdm(range(total_dialog), desc="dialogue"):
            sys_response = '' if self.user_agent.nlu else []
            random.seed(goal_seeds[0])
            np.random.seed(goal_seeds[0])
            torch.manual_seed(goal_seeds[0])
            goal_seeds.pop(0)
            sess.init_session()

            usr_da_list = []
            failed_da_sys = []
            failed_da_usr = []
            last_sys_da = None

            step = 0

            # print('init goal:',file=f)
            # # print(sess.evaluator.goal, file=f)
            # # pprint(sess.evaluator.goal)
            # print(sess.user_agent.policy.policy.goal.domain_goals, file=f)
            # print('-' * 50,file=f)

            for i in range(40):
                sys_response, user_response, session_over, reward = sess.next_turn(
                    sys_response)
                print('user in', sess.user_agent.get_in_da(),file=flog)
                print('user out', sess.user_agent.get_out_da(),file=flog)
                #
                # print('sys in', sess.sys_agent.get_in_da(),file=flog)
                # print('sys out', sess.sys_agent.get_out_da(),file=flog)
                print('user:', user_response,file=flog)
                print('sys:', sys_response,file=flog)

                step += 2

                if hasattr(sess.sys_agent, "get_in_da") and isinstance(sess.sys_agent.get_in_da(), list) \
                        and sess.user_agent.get_out_da() != [] \
                        and sess.user_agent.get_out_da() != sess.sys_agent.get_in_da():
                    for da1 in sess.user_agent.get_out_da():
                        for da2 in sess.sys_agent.get_in_da():
                            if da1 != da2 and da1 is not None and da2 is not None and (da1, da2) not in failed_da_sys:
                                failed_da_sys.append((da1, da2))

                if isinstance(last_sys_da, list) \
                        and last_sys_da is not None and last_sys_da != [] and sess.user_agent.get_in_da() != last_sys_da:
                    for da1 in last_sys_da:
                        for da2 in sess.user_agent.get_in_da():
                            if da1 != da2 and da1 is not None and da2 is not None and (da1, da2) not in failed_da_usr:
                                failed_da_usr.append((da1, da2))

                last_sys_da = sess.sys_agent.get_out_da() if hasattr(sess.sys_agent, "get_out_da") else None
                usr_da_list.append(sess.user_agent.get_out_da())

                if session_over:
                    break

            task_success = sess.evaluator.task_success()
            task_complete = sess.user_agent.policy.policy.goal.task_complete()
            book_rate = sess.evaluator.book_rate()
            stats = sess.evaluator.inform_F1()
            percentage = sess.evaluator.final_goal_analyze()
            if task_success:
                suc_num += 1
                turn_suc_num += step
            if task_complete:
                complete_num += 1
            if stats[2] is not None:
                precision.append(stats[0])
                recall.append(stats[1])
                f1.append(stats[2])
            if book_rate is not None:
                match.append(book_rate)
            if len(sess.evaluator.goal) > 0:
                num_domains += len(sess.evaluator.goal)
                num_domains_satisfying_constraints += len(sess.evaluator.goal) * percentage
            num_dialogs_satisfying_constraints += (percentage == 1)
            if (j+1) % 100 == 0:
                logger.info("model name %s", model_name)
                logger.info("dialogue %d", j+1)
                logger.info(sess.evaluator.goal)
                logger.info('task complete: %.3f', complete_num/(j+1))
                logger.info('task success: %.3f', suc_num/(j+1))
                logger.info('book rate: %.3f', np.mean(match))
                logger.info('inform precision/recall/f1: %.3f %.3f %.3f', np.mean(precision), np.mean(recall), np.mean(f1))
                logging.info("percentage of domains that satisfy the database constraints: %.3f" % \
                             (1 if num_domains == 0 else (num_domains_satisfying_constraints / num_domains)))
                logging.info("percentage of dialogs that satisfy the database constraints: %.3f" % (num_dialogs_satisfying_constraints / (j + 1)))
            domain_set = []
            for da in sess.evaluator.usr_da_array:
                if da.split('-')[0] != 'general' and da.split('-')[0] not in domain_set:
                    domain_set.append(da.split('-')[0])

            turn_num += step

            da_list = usr_da_list
            cycle_start = []
            for da in usr_da_list:
                if len(da) == 1 and da[0][2] == 'general':
                    continue

                if usr_da_list.count(da) > 1 and da not in cycle_start:
                    cycle_start.append(da)

            domain_turn = []
            for da in usr_da_list:
                if len(da) > 0 and da[0] is not None and len(da[0]) > 2:
                    domain_turn.append(da[0][1].lower())

            for domain in domain_set:
                reporter.record(domain, sess.evaluator.domain_success(domain), sess.evaluator.domain_reqt_inform_analyze(domain), failed_da_sys, failed_da_usr, cycle_start, domain_turn)

        tmp = 0 if suc_num == 0 else turn_suc_num / suc_num
        print("=" * 100)
        print("complete number of dialogs/tot:", complete_num / total_dialog)
        print("success number of dialogs/tot:", suc_num / total_dialog)
        print("average precision:", np.mean(precision))
        print("average recall:", np.mean(recall))
        print("average f1:", np.mean(f1))
        print('average book rate:', np.mean(match))
        print("average turn (succ):", tmp)
        print("average turn (all):", turn_num / total_dialog)
        print("percentage of domains that satisfy the database constraints: %.3f" % \
              (1 if num_domains == 0 else (num_domains_satisfying_constraints / num_domains)))
        print("percentage of dialogs that satisfy the database constraints: %.3f" % (num_dialogs_satisfying_constraints / total_dialog))
        print("=" * 100)
        print("complete number of dialogs/tot:", complete_num / total_dialog, file=f)
        print("success number of dialogs/tot:", suc_num / total_dialog, file=f)
        print("average precision:", np.mean(precision), file=f)
        print("average recall:", np.mean(recall), file=f)
        print("average f1:", np.mean(f1), file=f)
        print('average book rate:', np.mean(match), file=f)
        print("average turn (succ):", tmp, file=f)
        print("average turn (all):", turn_num / total_dialog, file=f)
        print("percentage of domains that satisfy the database constraints: %.3f" % \
              (1 if num_domains == 0 else (num_domains_satisfying_constraints / num_domains)), file=f)
        print("percentage of dialogs that satisfy the database constraints: %.3f" % (num_dialogs_satisfying_constraints / total_dialog), file=f)
        f.close()

        reporter.report(complete_num/total_dialog, suc_num/total_dialog, np.mean(precision), np.mean(recall), np.mean(f1), tmp, turn_num / total_dialog)

        return complete_num/total_dialog, suc_num/total_dialog, np.mean(precision), np.mean(recall), np.mean(f1), np.mean(match), turn_num / total_dialog

    def compare_models(self, agent_list, model_name, total_dialog=100):
        if len(agent_list) != len(model_name):
            return
        if len(agent_list) <= 0:
            return

        seed = random.randint(1, 100000)

        y0, y1, y2, y3, y4, y5, y6 = [], [], [], [], [], [], []
        for i in range(len(agent_list)):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            # print(model_name[i], total_dialog)
            complete, suc, pre, rec, f1, match, turn = self.comprehensive_analyze(agent_list[i], model_name[i], total_dialog)
            y0.append(complete)
            y1.append(suc)
            y2.append(pre)
            y3.append(rec)
            y4.append(f1)
            y5.append(match)
            y6.append(turn)

        x1 = list(range(1, 1 + len(model_name)))
        x1 = np.array(x1)
        x2 = x1 + 0.1
        x3 = x2 + 0.1
        x4 = x3 + 0.1

        plt.figure(figsize=(12, 7), dpi=300)

        font1 = {'weight': 'normal','size' : 20}

        font2 = {'weight': 'bold','size' : 22}

        font3 = {'weight': 'bold','size' : 35}
        plt.tick_params(axis='y', labelsize=20)
        plt.tick_params(axis='x', labelsize=22)
        plt.ylabel('score', font2)
        plt.ylim(0, 1)
        plt.xlabel('system', font2)
        plt.title('Comparison of different systems', font3, pad=16)

        plt.bar(x1, y0, width=0.1, align='center', label='Task complete')
        plt.bar(x2, y1, width=0.1, align='center', tick_label=model_name, label='Success rate')
        plt.bar(x3, y4, width=0.1, align='center', label='Inform F1')
        plt.bar(x4, y5, width=0.1, align='center', label='Book rate')
        plt.legend(loc=2, prop=font1)
        if not os.path.exists('results/'):
            os.mkdir('results')
        plt.savefig('results/compare_results.jpg')
        plt.close()
