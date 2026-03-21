"""
Registry of supported (dataset, model, task) configurations and class paths.

Adapted from SysEval-NegoLLMs.
"""

SUPPORTED_CONFIGS = set([
    # DND + OpenAI
    ("dnd", "open_ai", "sta_total_item_count_dnd"),
    ("dnd", "open_ai", "sta_max_points_dnd"),
    ("dnd", "open_ai", "mid_dial_act_dnd"),
    ("dnd", "open_ai", "mid_gen_resp_dnd"),
    ("dnd", "open_ai", "end_deal_specifics_dnd"),
    ("dnd", "open_ai", "sta_ask_point_values_dnd"),
    ("dnd", "open_ai", "mid_full_proposal_dnd"),
    ("dnd", "open_ai", "end_deal_total_dnd"),

    # DND + HF
    ("dnd", "hf_model", "sta_total_item_count_dnd"),
    ("dnd", "hf_model", "sta_max_points_dnd"),
    ("dnd", "hf_model", "mid_dial_act_dnd"),
    ("dnd", "hf_model", "mid_gen_resp_dnd"),
    ("dnd", "hf_model", "end_deal_specifics_dnd"),
    ("dnd", "hf_model", "sta_ask_point_values_dnd"),
    ("dnd", "hf_model", "mid_full_proposal_dnd"),
    ("dnd", "hf_model", "end_deal_total_dnd"),

    # Casino + OpenAI
    ("casino", "open_ai", "sta_total_item_count_ca"),
    ("casino", "open_ai", "mid_strategy_ca"),
    ("casino", "open_ai", "mid_gen_resp_ca"),
    ("casino", "open_ai", "end_deal_specifics_ca"),
    ("casino", "open_ai", "end_deal_total_ca"),
    ("casino", "open_ai", "sta_max_points_ca"),
    ("casino", "open_ai", "sta_ask_point_values_ca"),
    ("casino", "open_ai", "sta_ask_low_priority_ca"),
    ("casino", "open_ai", "sta_ask_high_priority_ca"),
    ("casino", "open_ai", "mid_ask_low_priority_ca"),
    ("casino", "open_ai", "mid_ask_high_priority_ca"),
    ("casino", "open_ai", "mid_partner_ask_low_priority_ca"),
    ("casino", "open_ai", "mid_partner_ask_high_priority_ca"),
    ("casino", "open_ai", "end_deal_likeness_ca"),
    ("casino", "open_ai", "end_deal_satisfaction_ca"),
    ("casino", "open_ai", "end_partner_deal_likeness_ca"),
    ("casino", "open_ai", "end_partner_deal_satisfaction_ca"),

    # Casino + HF
    ("casino", "hf_model", "sta_total_item_count_ca"),
    ("casino", "hf_model", "mid_strategy_ca"),
    ("casino", "hf_model", "mid_gen_resp_ca"),
    ("casino", "hf_model", "end_deal_specifics_ca"),
    ("casino", "hf_model", "end_deal_total_ca"),
    ("casino", "hf_model", "sta_max_points_ca"),
    ("casino", "hf_model", "sta_ask_point_values_ca"),
    ("casino", "hf_model", "sta_ask_low_priority_ca"),
    ("casino", "hf_model", "sta_ask_high_priority_ca"),
    ("casino", "hf_model", "mid_ask_low_priority_ca"),
    ("casino", "hf_model", "mid_ask_high_priority_ca"),
    ("casino", "hf_model", "mid_partner_ask_low_priority_ca"),
    ("casino", "hf_model", "mid_partner_ask_high_priority_ca"),
    ("casino", "hf_model", "end_deal_likeness_ca"),
    ("casino", "hf_model", "end_deal_satisfaction_ca"),
    ("casino", "hf_model", "end_partner_deal_likeness_ca"),
    ("casino", "hf_model", "end_partner_deal_satisfaction_ca"),

    # Job Interview + OpenAI
    ("job_interview", "open_ai", "end_deal_specifics_ji"),
    ("job_interview", "open_ai", "sta_ask_high_priority_ji_w"),
    ("job_interview", "open_ai", "sta_ask_low_priority_ji_w"),
    ("job_interview", "open_ai", "mid_ask_high_priority_ji_w"),
    ("job_interview", "open_ai", "mid_ask_low_priority_ji_w"),
    ("job_interview", "open_ai", "mid_partner_ask_high_priority_ji_w"),
    ("job_interview", "open_ai", "mid_partner_ask_low_priority_ji_w"),
    ("job_interview", "open_ai", "mid_dial_act_ji"),

    # Job Interview + HF
    ("job_interview", "hf_model", "end_deal_specifics_ji"),
    ("job_interview", "hf_model", "sta_ask_high_priority_ji_w"),
    ("job_interview", "hf_model", "sta_ask_low_priority_ji_w"),
    ("job_interview", "hf_model", "mid_ask_high_priority_ji_w"),
    ("job_interview", "hf_model", "mid_ask_low_priority_ji_w"),
    ("job_interview", "hf_model", "mid_partner_ask_high_priority_ji_w"),
    ("job_interview", "hf_model", "mid_partner_ask_low_priority_ji_w"),
    ("job_interview", "hf_model", "mid_dial_act_ji"),

    # CRA + OpenAI
    ("cra", "open_ai", "mid_dial_act_cra"),
    ("cra", "open_ai", "mid_full_proposal_cra"),

    # CRA + HF
    ("cra", "hf_model", "mid_dial_act_cra"),
    ("cra", "hf_model", "mid_full_proposal_cra"),

    # DND + Local
    ("dnd", "local_model", "sta_total_item_count_dnd"),
    ("dnd", "local_model", "sta_max_points_dnd"),
    ("dnd", "local_model", "mid_dial_act_dnd"),
    ("dnd", "local_model", "mid_gen_resp_dnd"),
    ("dnd", "local_model", "end_deal_specifics_dnd"),
    ("dnd", "local_model", "sta_ask_point_values_dnd"),
    ("dnd", "local_model", "mid_full_proposal_dnd"),
    ("dnd", "local_model", "end_deal_total_dnd"),

    # Casino + Local
    ("casino", "local_model", "sta_total_item_count_ca"),
    ("casino", "local_model", "mid_strategy_ca"),
    ("casino", "local_model", "mid_gen_resp_ca"),
    ("casino", "local_model", "end_deal_specifics_ca"),
    ("casino", "local_model", "end_deal_total_ca"),
    ("casino", "local_model", "sta_max_points_ca"),
    ("casino", "local_model", "sta_ask_point_values_ca"),
    ("casino", "local_model", "sta_ask_low_priority_ca"),
    ("casino", "local_model", "sta_ask_high_priority_ca"),
    ("casino", "local_model", "mid_ask_low_priority_ca"),
    ("casino", "local_model", "mid_ask_high_priority_ca"),
    ("casino", "local_model", "mid_partner_ask_low_priority_ca"),
    ("casino", "local_model", "mid_partner_ask_high_priority_ca"),
    ("casino", "local_model", "end_deal_likeness_ca"),
    ("casino", "local_model", "end_deal_satisfaction_ca"),
    ("casino", "local_model", "end_partner_deal_likeness_ca"),
    ("casino", "local_model", "end_partner_deal_satisfaction_ca"),

    # Job Interview + Local
    ("job_interview", "local_model", "end_deal_specifics_ji"),
    ("job_interview", "local_model", "sta_ask_high_priority_ji_w"),
    ("job_interview", "local_model", "sta_ask_low_priority_ji_w"),
    ("job_interview", "local_model", "mid_ask_high_priority_ji_w"),
    ("job_interview", "local_model", "mid_ask_low_priority_ji_w"),
    ("job_interview", "local_model", "mid_partner_ask_high_priority_ji_w"),
    ("job_interview", "local_model", "mid_partner_ask_low_priority_ji_w"),
    ("job_interview", "local_model", "mid_dial_act_ji"),

    # CRA + Local
    ("cra", "local_model", "mid_dial_act_cra"),
    ("cra", "local_model", "mid_full_proposal_cra"),

    # DND + vLLM
    ("dnd", "vllm_model", "sta_total_item_count_dnd"),
    ("dnd", "vllm_model", "sta_max_points_dnd"),
    ("dnd", "vllm_model", "mid_dial_act_dnd"),
    ("dnd", "vllm_model", "mid_gen_resp_dnd"),
    ("dnd", "vllm_model", "end_deal_specifics_dnd"),
    ("dnd", "vllm_model", "sta_ask_point_values_dnd"),
    ("dnd", "vllm_model", "mid_full_proposal_dnd"),
    ("dnd", "vllm_model", "end_deal_total_dnd"),

    # Casino + vLLM
    ("casino", "vllm_model", "sta_total_item_count_ca"),
    ("casino", "vllm_model", "mid_strategy_ca"),
    ("casino", "vllm_model", "mid_gen_resp_ca"),
    ("casino", "vllm_model", "end_deal_specifics_ca"),
    ("casino", "vllm_model", "end_deal_total_ca"),
    ("casino", "vllm_model", "sta_max_points_ca"),
    ("casino", "vllm_model", "sta_ask_point_values_ca"),
    ("casino", "vllm_model", "sta_ask_low_priority_ca"),
    ("casino", "vllm_model", "sta_ask_high_priority_ca"),
    ("casino", "vllm_model", "mid_ask_low_priority_ca"),
    ("casino", "vllm_model", "mid_ask_high_priority_ca"),
    ("casino", "vllm_model", "mid_partner_ask_low_priority_ca"),
    ("casino", "vllm_model", "mid_partner_ask_high_priority_ca"),
    ("casino", "vllm_model", "end_deal_likeness_ca"),
    ("casino", "vllm_model", "end_deal_satisfaction_ca"),
    ("casino", "vllm_model", "end_partner_deal_likeness_ca"),
    ("casino", "vllm_model", "end_partner_deal_satisfaction_ca"),

    # Job Interview + vLLM
    ("job_interview", "vllm_model", "end_deal_specifics_ji"),
    ("job_interview", "vllm_model", "sta_ask_high_priority_ji_w"),
    ("job_interview", "vllm_model", "sta_ask_low_priority_ji_w"),
    ("job_interview", "vllm_model", "mid_ask_high_priority_ji_w"),
    ("job_interview", "vllm_model", "mid_ask_low_priority_ji_w"),
    ("job_interview", "vllm_model", "mid_partner_ask_high_priority_ji_w"),
    ("job_interview", "vllm_model", "mid_partner_ask_low_priority_ji_w"),
    ("job_interview", "vllm_model", "mid_dial_act_ji"),

    # CRA + vLLM
    ("cra", "vllm_model", "mid_dial_act_cra"),
    ("cra", "vllm_model", "mid_full_proposal_cra"),
])


CLS_NAME2PATHS = {
    "datasets": {
        "dnd": "eval.datasets.dealornodeal.DNDHandler",
        "casino": "eval.datasets.casino.CasinoHandler",
        "job_interview": "eval.datasets.jobinterview.JIHandler",
        "cra": "eval.datasets.cra.CRAHandler",
    },
    "models": {
        "open_ai": "eval.models.openai_model.OpenAIHandler",
        "hf_model": "eval.models.hf_model.HFModelHandler",
        "local_model": "eval.models.local_model.LocalModelHandler",
        "vllm_model": "eval.models.vllm_model.VLLMModelHandler",
    },
    "tasks": {
        "sta_total_item_count_dnd": "eval.tasks.sta_total_item_count_dnd.TICNDHandlerDND",
        "sta_max_points_dnd": "eval.tasks.sta_max_points_dnd.A1MPNDHandlerDND",
        "mid_dial_act_dnd": "eval.tasks.mid_dial_act_dnd.DASUHandler",
        "mid_gen_resp_dnd": "eval.tasks.mid_gen_resp_dnd.GSDNDHandler",
        "end_deal_specifics_dnd": "eval.tasks.end_deal_specifics_dnd.A1BCHandler",
        "sta_ask_point_values_dnd": "eval.tasks.sta_ask_point_values_dnd.BYDNDNDPointValuesHandler",
        "mid_full_proposal_dnd": "eval.tasks.mid_full_proposal_dnd.DNDRegAllSlotsHandler",
        "end_deal_total_dnd": "eval.tasks.end_deal_total_dnd.YDNDDealPointsHandler",

        "sta_total_item_count_ca": "eval.tasks.sta_total_item_count_ca.TICHandlerCa",
        "mid_strategy_ca": "eval.tasks.mid_strategy_ca.NSUHandler",
        "mid_gen_resp_ca": "eval.tasks.mid_gen_resp_ca.GSCaHandler",
        "end_deal_specifics_ca": "eval.tasks.end_deal_specifics_ca.A1FCHandler",
        "end_deal_total_ca": "eval.tasks.end_deal_total_ca.A1PHandlerCa",
        "sta_max_points_ca": "eval.tasks.sta_max_points_ca.A1CaNDMaxPointsHandler",
        "sta_ask_point_values_ca": "eval.tasks.sta_ask_point_values_ca.Food1CaNDPointValuesHandler",
        "sta_ask_low_priority_ca": "eval.tasks.sta_ask_low_priority_ca.Low1CaWCPrioritiesHandler",
        "sta_ask_high_priority_ca": "eval.tasks.sta_ask_high_priority_ca.High1CaWCPrioritiesHandler",
        "mid_ask_low_priority_ca": "eval.tasks.mid_ask_low_priority_ca.MidLow1CaWCPrioritiesHandler",
        "mid_ask_high_priority_ca": "eval.tasks.mid_ask_high_priority_ca.MidHigh1CaWCPrioritiesHandler",
        "mid_partner_ask_low_priority_ca": "eval.tasks.mid_partner_ask_low_priority_ca.MidPartnerLow1CaWCPrioritiesHandler",
        "mid_partner_ask_high_priority_ca": "eval.tasks.mid_partner_ask_high_priority_ca.MidPartnerHigh1CaWCPrioritiesHandler",
        "end_deal_likeness_ca": "eval.tasks.end_deal_likeness_ca.A1LikenessCAHandler",
        "end_deal_satisfaction_ca": "eval.tasks.end_deal_satisfaction_ca.A1SatisfactionCAHandler",
        "end_partner_deal_likeness_ca": "eval.tasks.end_partner_deal_likeness_ca.A1PartnerLikenessCAHandler",
        "end_partner_deal_satisfaction_ca": "eval.tasks.end_partner_deal_satisfaction_ca.A1PartnerSatisfactionCAHandler",

        "end_deal_specifics_ji": "eval.tasks.end_deal_specifics_ji.FComHandler",
        "sta_ask_high_priority_ji_w": "eval.tasks.sta_ask_high_priority_ji_w.WHighJIPrioritiesHandler",
        "sta_ask_low_priority_ji_w": "eval.tasks.sta_ask_low_priority_ji_w.WLowJIPrioritiesHandler",
        "mid_ask_high_priority_ji_w": "eval.tasks.mid_ask_high_priority_ji_w.WMidHighJIPrioritiesHandler",
        "mid_ask_low_priority_ji_w": "eval.tasks.mid_ask_low_priority_ji_w.WMidLowJIPrioritiesHandler",
        "mid_partner_ask_high_priority_ji_w": "eval.tasks.mid_partner_ask_high_priority_ji_w.WMidPartnerHighJIPrioritiesHandler",
        "mid_partner_ask_low_priority_ji_w": "eval.tasks.mid_partner_ask_low_priority_ji_w.WMidPartnerLowJIPrioritiesHandler",
        "mid_dial_act_ji": "eval.tasks.mid_dial_act_ji.JIRegDAHandler",

        "mid_dial_act_cra": "eval.tasks.mid_dial_act_cra.CRARegDAHandler",
        "mid_full_proposal_cra": "eval.tasks.mid_full_proposal_cra.CRARegAllSlotsHandler",
    },
}


TASK_TO_DATASET = {}
for (dataset, model, task) in SUPPORTED_CONFIGS:
    TASK_TO_DATASET.setdefault(task, dataset)


def get_tasks_for_dataset(dataset_name):
    """Return all task names available for a given dataset."""
    return sorted({t for (d, m, t) in SUPPORTED_CONFIGS if d == dataset_name})


def get_all_task_names():
    """Return all known task names."""
    return sorted(CLS_NAME2PATHS["tasks"].keys())
