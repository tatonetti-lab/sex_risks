from select_patients import fetch_patients
from psm_make_feature import make_features_mp
from psm_model_rf import run_rf_model
from calculate_propensity_scores import calculate_propensity_scores
from run import run_analysis_mp
from results_make import make_results
from results_compile import compile_results
from important_sex_biased_adr import filter_sex_risks

if __name__ == '__main__':
    print("Fetching patients")
    #fetch_patients()

    print("Make features")
    #make_features_mp()

    print("Running propensity score matching")
    #run_rf_model()

    print("Calculate propensity scores")
    #calculate_propensity_scores()

    print("Run analysis")
    run_analysis_mp()

    print("Making results")
    make_results()

    print("Compiling results")
    compile_results() 

    print("Filtering sex risks")
    filter_sex_risks() 
