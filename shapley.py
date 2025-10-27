
import pickle
import shap
import pandas as pd



def scoring_fn(X_df):
    return (
        (0.3 * X_df['region_score'].values) +
        (0.45 * X_df['travel_score'].values) +
        (0.25 * X_df['departure_score'].values)
    )

if __name__ == "__main__":
    df = pd.read_csv('visualization/final_region_travel_departure_poly2vec_mean_region_test_visit_scores.csv')
    df['normalized_region_score'] = (df['region_score'] - df['region_score'].min()) / (df['region_score'].max() - df['region_score'].min())
    df['normalized_travel_score'] = (df['travel_score'] - df['travel_score'].min()) / (df['travel_score'].max() - df['travel_score'].min())
    df['normalized_departure_score'] = (df['departure_score'] - df['departure_score'].min()) / (df['departure_score'].max() - df['departure_score'].min())


    X = df[['region_score', 'travel_score', 'departure_score']]
    explainer = shap.Explainer(scoring_fn, X)
    shap_values = explainer(X)

    # Save explainer
    with open('visualization/normalized_weighted_final_poly2vec_region_travel_departure_explainer.pkl', 'wb') as f:
        pickle.dump(explainer, f)

    # Save shap values
    with open('visualization/normalized_weighted_final_poly2vec_region_travel_departure_shap.pkl', 'wb') as f:
        pickle.dump(shap_values, f)