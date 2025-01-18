from main import score_calculator
import pandas as pd, numpy as np
import time, ast, statistics, os
import matplotlib.pyplot as plt

def helper():
    genai = pd.read_csv("Data/Combined/genAI_combined.csv", index_col="index")
    trad = pd.read_csv("Data/Combined/traditional_combined.csv", index_col="index")
    
    if not os.path.exists('./plots/explore/'):
        os.makedirs('./plots/explore/')
    
    final_data_genai = final_medians_plots(genai)
    final_data_trad = final_medians_plots(trad)
    create_chart_imedians(final_data_genai, final_data_trad)
    
    genai_fpr = final_ifpr2(genai)
    trad_fpr = final_ifpr2(trad)
    create_chart_ifpr(genai_fpr, trad_fpr, ext="ifpr")
    
def final_ifpr2(input_data):
    start = time.time()
    identities = ['women', 'lgbt-related', 'disability', 'non-white', 'christian', 'non-christian', 'white', 'men', 'straight', "none"]
    data = input_data[["OpenAI_ME_responses", "true_label", "Big_identity"]].copy()
    data['OpenAI_ME_responses'] = data['OpenAI_ME_responses'].apply(lambda x: ast.literal_eval(x)['results'][0]['categories'])

    # Initialize dictionaries
    categories = list(data.iloc[0]['OpenAI_ME_responses'].keys())
    overall_ifpr_dict = {x: [] for x in categories if '/' not in x}
    identity_ifpr_dict = {identity: {x: {"true": [], "pred": []} for x in overall_ifpr_dict} for identity in identities}
    
    # Populate dictionaries
    def process_row(row):
        for score in overall_ifpr_dict:
            overall_ifpr_dict[score].append(row["OpenAI_ME_responses"][score])
        
        sub_ids = row['Big_identity']
        sub_ids = ["none"] if pd.isna(sub_ids) else sub_ids.lower().split(';')
        
        for identity in sub_ids:
            for score in overall_ifpr_dict:
                identity_ifpr_dict[identity][score]['pred'].append(row["OpenAI_ME_responses"][score])
                identity_ifpr_dict[identity][score]['true'].append(row["true_label"])

    data.apply(process_row, axis=1)
    
    # Calculate overall FPR
    overall_ifpr_dict = {score: score_calculator({"true": data["true_label"].tolist(), "pred": preds}, score="FPR") for score, preds in overall_ifpr_dict.items()}
    
    # Calculate identity-specific FPR relative to overall FPR
    for identity, scores in identity_ifpr_dict.items():
        for score in scores:
            identity_ifpr_dict[identity][score] = score_calculator(scores[score], score="FPR") / overall_ifpr_dict[score]
    
    print("Time taken:", round(time.time()-start, 3))
    return {"OpenAI": identity_ifpr_dict}

def final_medians_plots(input_data):
    input_data = input_data[input_data["true_label"] == 0]
    score_cols = ['OpenAI_normalized', 'Google_ME_responses']
    data = input_data[score_cols + ['OpenAI_ME_bool', 'Big_identity']]
    
    data['Big_identity'] = data['Big_identity'].apply(lambda x: x.lower().split(';') if type(x) != float else ["none"])
    data['OpenAI_normalized'] = data['OpenAI_normalized'].apply(lambda x: ast.literal_eval(x))
    data['Google_ME_responses'] = data['Google_ME_responses'].apply(lambda x: ast.literal_eval(x))
    
    final_data = {}
    numpy_data = data[['OpenAI_normalized', 'Google_ME_responses', 'Big_identity']].to_numpy()
    plot_data = imedians_dict_combined(numpy_data)
    boot_strapped_data = bootstrap_category_imedians(numpy_data, n=1000)
    
    for col in score_cols:
        me = col.split("_")[0]
        final_data[me] = [plot_data[me], boot_strapped_data[me]]
    
    return final_data


def imedians_dict_combined(data):
    identities = ['women', 'lgbt-related', 'disability', 'non-white', 'christian', 'non-christian', 'white', 'men', 'straight', "none"]
    score_dict = {identity: {score: [] for score in data[0][0] if '/' not in score} for identity in identities}
    score_dict_google = {identity: {score: [] for score in data[0][1]} for identity in identities}
    overall_openAI = {score: [] for score in data[0][0]}
    overall_google = {score: [] for score in data[0][1]}
    
    for row in data:
        for openAI_score in row[0]: overall_openAI[openAI_score].append(row[0][openAI_score])
        for google_score in row[1]: overall_google[google_score].append(row[1][google_score])
    
    for openAI_score in overall_openAI: overall_openAI[openAI_score] = statistics.median(overall_openAI[openAI_score])
    for google_score in overall_google: overall_google[google_score] = statistics.median(overall_google[google_score])
    
    max_median_google = max(overall_google.values())
    max_median_openai = max(overall_openAI.values())
    
    for row in data:
        for score in row[0].keys():
            if '/' in score: continue
            for sub_id in row[2]:
                score_dict[sub_id][score].append(row[0][score])
        for score in row[1].keys():
            for sub_id in row[2]:
                score_dict_google[sub_id][score].append(row[1][score])

    for identity in identities:
        for score in score_dict[identity]:
            score_dict[identity][score] = statistics.median(score_dict[identity][score])/max_median_openai
        for score in score_dict_google[identity]:
            score_dict_google[identity][score] = statistics.median(score_dict_google[identity][score])/max_median_google
            
    return {'OpenAI': score_dict, 'Google': score_dict_google}

def bootstrap_category_imedians(data: np.ndarray, n: int=1000):
    full_identities_dict_openai = {}
    full_identities_dict_google = {}
    
    for _ in range(n):
        print(_)
        identities_dict_openai = {}
        identities_dict_google = {}
        
        boot_strapped_data = data[np.random.choice(data.shape[0], data.shape[0], replace=True)]
        overall_openAI = {score: [] for score in data[0][0]}
        overall_google = {score: [] for score in data[0][1]}
        for openai_scores, google_scores, identities in boot_strapped_data:
            for openAI_score in openai_scores: overall_openAI[openAI_score].append(openai_scores[openAI_score])
            for google_score in google_scores: overall_google[google_score].append(google_scores[google_score])
            for identity in identities:
                if identity not in identities_dict_openai:
                    identities_dict_openai[identity] = {'scores': {key: [] for key in openai_scores.keys()}, 'count': 0}
                for key in openai_scores.keys():
                    identities_dict_openai[identity]['scores'][key].append(openai_scores[key])
                identities_dict_openai[identity]['count'] += 1
                
                if identity not in identities_dict_google:
                    identities_dict_google[identity] = {'scores': {key: [] for key in google_scores.keys()}, 'count': 0}
                for key in google_scores.keys():
                    identities_dict_google[identity]['scores'][key].append(google_scores[key])
                identities_dict_google[identity]['count'] += 1
        for openAI_score in overall_openAI: overall_openAI[openAI_score] = statistics.median(overall_openAI[openAI_score])
        for google_score in overall_google: overall_google[google_score] = statistics.median(overall_google[google_score])
        
        max_median_google = max(overall_google.values())
        max_median_openai = max(overall_openAI.values())
        
        for identity in identities_dict_openai:
            median_scores_openai = {key: statistics.median(identities_dict_openai[identity]['scores'][key]) for key in identities_dict_openai[identity]['scores']}
            if identity not in full_identities_dict_openai:
                full_identities_dict_openai[identity] = {key: [] for key in median_scores_openai}
            for key in median_scores_openai:
                full_identities_dict_openai[identity][key].append(round(median_scores_openai[key]/max_median_openai, 3))
        
        for identity in identities_dict_google:
            median_scores_google = {key: statistics.median(identities_dict_google[identity]['scores'][key]) for key in identities_dict_google[identity]['scores']}
            if identity not in full_identities_dict_google:
                full_identities_dict_google[identity] = {key: [] for key in median_scores_google}
            for key in median_scores_google:
                full_identities_dict_google[identity][key].append(round(median_scores_google[key]/max_median_google, 3))
    
    for identity in full_identities_dict_openai:
        for key in full_identities_dict_openai[identity]:
            full_identities_dict_openai[identity][key].sort()
            ci_upper = np.quantile(full_identities_dict_openai[identity][key], 0.975)
            ci_lower = np.quantile(full_identities_dict_openai[identity][key], 0.025)
            full_identities_dict_openai[identity][key] = [ci_upper, ci_lower]

    for identity in full_identities_dict_google:
        for key in full_identities_dict_google[identity]:
            full_identities_dict_google[identity][key].sort()
            ci_upper = np.quantile(full_identities_dict_google[identity][key], 0.975)
            ci_lower = np.quantile(full_identities_dict_google[identity][key], 0.025)
            full_identities_dict_google[identity][key] = [ci_upper, ci_lower]
    
    return {'OpenAI': full_identities_dict_openai, 'Google': full_identities_dict_google}

def create_chart_ifpr(final_data_genai, final_data_trad, ext=""):
    for api in final_data_genai.keys():
        
        plot_data_genai = final_data_genai[api]
        plot_data_trad = final_data_trad[api]

        identities = list(plot_data_genai.keys())
        scores = list(plot_data_genai[identities[0]].keys())

        for identity in identities:
            genai_values = [plot_data_genai[identity][score] for score in scores]
            trad_values = [plot_data_trad[identity][score] for score in scores]

            x = np.arange(len(scores))
            width = 0.35

            fig, ax = plt.subplots(figsize=(15, 10))

            ax.bar(x - width/2, genai_values, width, label='GenAI', capsize=5, color='skyblue')
            ax.bar(x + width/2, trad_values, width, label='Traditional', capsize=5, color='lightgreen')

            
            ax.set_xlabel('Scores')
            ax.set_ylabel('Values')
            ax.set_title(f'{api} - {identity} Scores with Confidence Intervals')
            ax.set_xticks(x)
            ax.set_xticklabels(scores, rotation=45, ha='right')
            ax.legend()

            plt.axhline(y=1.0, color='black', linestyle='-')
            plt.savefig(f"./plots/explore/test/{ext}_{api}_{identity}_scores.png", pad_inches=1, bbox_inches='tight')
            plt.clf()


def create_chart_imedians(final_data_genai, final_data_trad, ext=""):
    for api in final_data_genai.keys():
        plot_data_genai, boot_strapped_data_genai = final_data_genai[api]
        plot_data_trad, boot_strapped_data_trad = final_data_trad[api]
        
        identities = list(plot_data_genai.keys())
        scores = list(plot_data_genai[identities[0]].keys())

        for identity in identities:
            genai_values = [plot_data_genai[identity][score] for score in scores]
            trad_values = [plot_data_trad[identity][score] for score in scores]

            genai_ci_upper = [boot_strapped_data_genai[identity][score][0] for score in scores]
            genai_ci_lower = [boot_strapped_data_genai[identity][score][1] for score in scores]

            trad_ci_upper = [boot_strapped_data_trad[identity][score][0] for score in scores]
            trad_ci_lower = [boot_strapped_data_trad[identity][score][1] for score in scores]

            yerr_genai = [[max(0, v - l) for v, l in zip(genai_values, genai_ci_lower)], 
                          [max(0, u - v) for v, u in zip(genai_values, genai_ci_upper)]]

            yerr_trad = [[max(0, v - l) for v, l in zip(trad_values, trad_ci_lower)], 
                         [max(0, u - v) for v, u in zip(trad_values, trad_ci_upper)]]

            x = np.arange(len(scores))
            width = 0.35

            fig, ax = plt.subplots(figsize=(15, 10))

            ax.bar(x - width/2, genai_values, width, label='GenAI', yerr=yerr_genai, capsize=5, color='skyblue')
            ax.bar(x + width/2, trad_values, width, label='Traditional', yerr=yerr_trad, capsize=5, color='lightgreen')
            
            ax.set_xlabel('Scores')
            ax.set_ylabel('Values')
            ax.set_title(f'{api} - {identity} Scores with Confidence Intervals')
            ax.set_xticks(x)
            ax.set_xticklabels(scores, rotation=45, ha='right')
            ax.legend()

            plt.axhline(y=1.0, color='black', linestyle='-')
            plt.savefig(f"./plots/explore/test/{ext}_{api}_{identity}_scores.png", pad_inches=1, bbox_inches='tight')
            plt.clf()
         
            
if __name__ == "__main__":
    helper()