from convlab2.util.analysis_tool.analyzer import Analyzer
from convlab2.nlu.jointBERT.multiwoz import BERTNLU
from convlab2.nlu.svm.multiwoz import SVMNLU
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.nlg.template.multiwoz import TemplateNLG
from convlab2.dialog_agent import PipelineAgent


def build_user_agent_bertnlu():
    user_nlu = BERTNLU()
    user_dst = None
    user_policy = RulePolicy(character='usr')
    user_nlg = TemplateNLG(is_user=True)
    user_agent = PipelineAgent(user_nlu, user_dst, user_policy, user_nlg, 'user')
    return user_agent


def build_sys_agent_bertnlu():
    sys_nlu = BERTNLU()
    sys_dst = RuleDST()
    sys_policy = RulePolicy(character='sys')
    sys_nlg = TemplateNLG(is_user=False)
    sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, 'sys')
    return sys_agent


def build_sys_agent_svmnlu():
    sys_nlu = SVMNLU()
    sys_dst = RuleDST()
    sys_policy = RulePolicy(character='sys')
    sys_nlg = TemplateNLG(is_user=False)
    sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, 'sys')
    return sys_agent


if __name__ == "__main__":
    # user agent for simulator
    user_agent = build_user_agent_bertnlu()

    # build your own sys agent, modify the func to change the settings
    sys_agent_svm = build_sys_agent_svmnlu()
    sys_agent_bert = build_sys_agent_bertnlu()

    # build analyzer, temporarily only for multiwoz
    analyzer = Analyzer(user_agent=user_agent, dataset='multiwoz')

    #sample dialog
    analyzer.sample_dialog(sys_agent_bert)

    #analyze and generate test report
    analyzer.comprehensive_analyze(sys_agent=sys_agent_svm, model_name='svmnlu', total_dialog=10)
    # analyzer.comprehensive_analyze(sys_agent=sys_agent_bert, model_name='bertnlu', total_dialog=100)

    #compare multiple model
    analyzer.compare_models(agent_list=[sys_agent_svm, sys_agent_bert], model_name=['svmnlu', 'bertnlu'], total_dialog=10)
