# Unified Decision Variable Space for EnergyOptAI Multi-Objective Optimization

DECISION_SPACE = {
    "feed_rate": {
        "symbol": "f",
        "unit": "mm/rev",
        "kaggle_col": "f",
        "mendeley_col": "F_val",
        "bounds": [0.07, 0.13],
        "description": "Feed rate per revolution"
    },
    "depth_of_cut": {
        "symbol": "ap",
        "unit": "mm",
        "kaggle_col": "ap",
        "mendeley_col": None,
        "bounds": [0.25, 0.80],
        "description": "Depth of cut"
    },
    "spindle_speed": {
        "symbol": "S",
        "unit": "rpm",
        "kaggle_col": None,
        "mendeley_col": "S",
        "bounds": [3000.0, 13000.0],
        "description": "Spindle motor rotation speed"
    },
    "tool_condition": {
        "symbol": "TCond",
        "unit": "mm",
        "role": "state_variable",
        "kaggle_col": "TCond",
        "mendeley_col": None,
        "bounds": [0.0, 0.23],
        "optimization_treatment": "scenario",
        "scenarios": {
            "new_tool": 0.0,
            "mid_life": 0.053,
            "worn_tool": 0.10
        },
        "note": "TCond treated as a scenario parameter, not a free decision variable. Optimization is run separately for each tool state."
    }
}
