import pandas as pd
import wikipedia as wiki
import numpy as np
import ast, os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from Movies.movie_tagger import titles_recursively
import statistics
import matplotlib.patches as mpatches
from scipy.stats import bootstrap


def add_identity_cols_combined_data():
    files = ['Data/Combined/genAI_combined.csv', 'Data/Combined/traditional_combined.csv']
    # files = ['Data/Markov/data_with_ME.csv']
    for file in files:
        data = pd.read_csv(file).reset_index()
        for identity_col in ['Big_identity', 'Sub_Identities']:
            identites = data[identity_col].tolist()
            for index, identity_tags in enumerate(identites):
                if type(identity_tags) != str:
                    continue
                identity_split = identity_tags.lower().split(';')
                for identity in identity_split:
                    if identity != 'other':
                        data.loc[index, identity] = 1
                    # all_identities.add(identity)
        data.to_csv(file, index=False)
        print(f'saved {file}')             


def add_has_slurs():
    files = ['Data/Markov/data_with_ME.csv', 'Data/Stormfront/stormfront_data_ME.csv',
             'Data/Tweets/offensive_twitter_data_with_ME.csv', 'Data/Tweets/hate_twitter_data_with_ME.csv']
    cols = ['prompt', 'Sentence', 'Text', 'Text']
    wiki_file = pd.read_csv("Data/wiki_identity.csv")
    slurs_list = {}
    for category, identity, terms in zip(wiki_file["Identity category"], wiki_file["Sub_identity"], wiki_file["Terms"]):
        terms = ast.literal_eval(terms)
        for term in terms:
            slurs_list[term] = [category, identity]
    
    for file, col in zip(files, cols):
        data = pd.read_csv(file)
        has_slurs = []
        for prompt in data[col].tolist():
            has_slur = False
            for key in slurs_list.keys():
                if f" {key.lower()} " in prompt.lower() \
                    or f" {key.lower()}s " in prompt.lower()\
                    or f" {key.lower()}es " in prompt.lower():
                    has_slur = True
                    break
            has_slurs.append(int(has_slur))
        data['has_slur'] = has_slurs
        
        data.to_csv(file, index=False)


def get_max_score(score_dict):
    max_ifpr = ['', 0, 0]
    for identity in score_dict:
        if max_ifpr[0] == '':
            max_ifpr[0] = identity
            max_ifpr[1] = score_dict[identity][0]
            max_ifpr[2] = score_dict[identity][1]
        else:
            if score_dict[identity][0] > max_ifpr[1]:
                max_ifpr[0] = identity
                max_ifpr[1] = score_dict[identity][0]
                max_ifpr[2] = score_dict[identity][1]

    return max_ifpr
    
            
def identity_imedians(dataset_path:str, label_cols:list, dataset_name:str, score_types:list, data_subset:str='', other_table=False, identity_type='big') -> dict:
    save_path = 'Data/imed_table.csv' if identity_type == 'big' else 'Data/imed_table_sub_identities.csv'
    other_save_path = f"Data/{dataset_name}.csv" if identity_type == 'big' else f"Data/{dataset_name}_sub_ids.csv"
    if len(score_types) != len(label_cols):
        print("Score and labels lists don't match")
        return
    
    ME_info = {'Google': 'Google_ME_responses', 'Perspective': 'perspective_ME_responses', 'OpenAI': 'OpenAI_normalized'}
    return_dict = {}
    categories = get_movies_categories('big')
    
    for label_col, score_type in zip(label_cols, score_types):
        full_data = pd.read_csv(dataset_path)
        
        # dixon toxic column not binary
        if "Dixon" in dataset_path:
            full_data[label_col] = full_data[label_col].apply(lambda x: 0 if x == 'nontoxic' else 1)
        full_data = full_data[full_data[label_col] == 0]

        max_imedians_dict = {}
        return_dict[label_col] = {}
        for ME, me_score_col in zip(ME_info.keys(), ME_info.values()):
            print(ME)
            data = full_data.copy()
            # getting scores
            if 'perspective' in me_score_col:
                data[me_score_col] = data[me_score_col].apply(lambda x: {"Toxic": ast.literal_eval(x)\
                    ["attributeScores"]["TOXICITY"]["spanScores"][0]["score"]["value"]}\
                        if "API error" not in x and "languageNotSupportedByAttributeError" not in x else {})
                data.drop(data[data[me_score_col] == {}].index, inplace=True)
            else:
                data[me_score_col] = data[me_score_col].apply(lambda x: ast.literal_eval(x)\
                    if "API error" not in x else {})
                data.drop(data[data[me_score_col] == {}].index, inplace=True)
            
            # converting dict of scores to max values
            data[me_score_col] = data[me_score_col].apply(lambda x: max(x.values())).tolist()
            overall_median_dict = {'scores': [], 'count': 0}
            if identity_type == 'big':
                id_col = 'Big_identity'
                identities_dict = {identity.lower() : {'scores' : [], 'count' : 0} for category in categories for identity in categories[category]}
            else:
                id_col = 'Sub_Identities'
                identities_dict = {}
            data_identities = data[id_col].tolist()

            for identities, score in zip(data_identities, data[me_score_col]):
                if type(identities) == float:
                    continue
                identities = identities.lower().split(';')
                for identity in identities:
                    if identity not in identities_dict:
                        identities_dict[identity] = {'scores' : [], 'count' : 0}
                    identities_dict[identity]['scores'].append(score)
                    identities_dict[identity]['count'] += 1
                overall_median_dict['count'] += 1
                overall_median_dict['scores'].append(score)

            overall_median = statistics.median(overall_median_dict['scores'])
            identities_dict = {identity: [round(statistics.median(identities_dict[identity]['scores'])/ overall_median if identities_dict[identity]['count'] > 0 else 0, 2),\
                identities_dict[identity]['count']] for identity in identities_dict}
            bootstrap_data = bootstrap_imedians(data, id_col, me_score_col)
            for bootstrap_identity in bootstrap_data:
                identities_dict[bootstrap_identity].append(bootstrap_data[bootstrap_identity])
            max_imedians_dict[ME] = get_max_score(identities_dict)
            return_dict[label_col][ME] = identities_dict
            
        if not other_table:
            row = {r"\textbf{Dataset}": dataset_name,
                    r"\textbf{Data subset}": data_subset,
                    r"\textbf{Score type}": score_type
                    }
            
            for ME, imed_data in zip(max_imedians_dict.keys(), max_imedians_dict.values()):
                row.update({
                    r'\textbf{ ' + ME + ' imedianscore}' : imed_data[1],
                    r'\textbf{ ' + ME + ' argmax}' : imed_data[0],
                    r'\textbf{ ' + ME + ' TN}' : imed_data[2],
                })
            try:
                table = pd.read_csv(save_path)
                table = pd.concat([table, pd.DataFrame(row, index=[0])])
            except Exception as e:
                table = pd.DataFrame(row, index=[0])
            
            table.drop_duplicates(inplace=True, subset=[r"\textbf{Dataset}", r'\textbf{Data subset}', r"\textbf{Score type}"], keep="last")
            table.to_csv(save_path, index=False)
            
    
    if other_table:
        df_dict = {}
        for label in return_dict:
            for ME in return_dict[label]:
                for identity in return_dict[label][ME]:
                    if identity not in df_dict:
                        df_dict[identity] = {r'\textbf{Identity group}' : identity}
                    df_dict[identity].update({
                        r'\textbf{ ' + ME + ' imedianscore}' : return_dict[label][ME][identity][0],
                        r'\textbf{ ' +  ME + ' TN}' : return_dict[label][ME][identity][1],
                        r'\textbf{ ' + ME + ' imedian CI}' : return_dict[label][ME][identity][2],
                    })

        try:
            table = pd.read_csv(other_save_path)
            table = pd.merge(table, pd.DataFrame(list(df_dict.values())), on=[r'\textbf{Identity group}'])

        except Exception as e:
            table = pd.DataFrame(df_dict.values())
        
        table.to_csv(other_save_path, index=False)
    
    return return_dict


def bootstrap_imedians(data: pd.DataFrame, id_col, score_col, n: int=1000):
    full_identities_dict = {}
    for _ in range(n):
        identities_dict = {}
        boot_strapped_data = data.sample(n=data.shape[0], replace=True)
        overall_median_dict = {'scores': [], 'count': 0}
        for identities, score in zip(boot_strapped_data[id_col], boot_strapped_data[score_col]):
            if type(identities) == float:
                continue
            identities = identities.lower().split(';')
            for identity in identities:
                if identity not in identities_dict:
                    identities_dict[identity] = {'scores': [], 'count': 0}
                identities_dict[identity]['scores'].append(score)
                identities_dict[identity]['count'] += 1
            overall_median_dict['count'] += 1
            overall_median_dict['scores'].append(score)
        overall_median = statistics.median(overall_median_dict['scores'])
        for identity in identities_dict:
            if identities_dict[identity]['count'] > 0:
                median_score = statistics.median(identities_dict[identity]['scores']) / overall_median
            else:
                median_score = 0
            if identity not in full_identities_dict:
                full_identities_dict[identity] = [round(median_score, 2)]
            else:
                full_identities_dict[identity].append(round(median_score,2))
            identities_dict[identity]['scores'].append(round(median_score, 2))

    for identity in full_identities_dict:
        full_identities_dict[identity].sort()
        full_identities_dict[identity] = [full_identities_dict[identity][int(len(full_identities_dict[identity])*0.975-1)],
                                                             full_identities_dict[identity][int(len(full_identities_dict[identity])*0.025)-1]]
    return full_identities_dict


def bootstrap_ifprs(data: pd.DataFrame, id_col, flag_col, label_col, n: int=1000):
    full_identities_dict = {}
    for _ in range(n):
        boot_strapped_data = data.sample(n=data.shape[0], replace=True)
        overall_fpr = score_calculator({'true' : boot_strapped_data[label_col], 'pred': boot_strapped_data[flag_col]}, score='FPR')
        identities_dict = {}
        for identities, label, flag in zip(boot_strapped_data[id_col], boot_strapped_data[label_col], boot_strapped_data[flag_col]):
            if type(identities) == float:
                continue
            identities = identities.lower().split(';')
            for identity in identities:
                if identity not in identities_dict:
                    identities_dict[identity] = {'true' : [], 'pred' : []}
                identities_dict[identity]['true'].append(label)
                identities_dict[identity]['pred'].append(flag)
        identities_dict = {identity : round(score_calculator(identities_dict[identity], score='FPR')/overall_fpr, 2) for identity in identities_dict}
        for identity in identities_dict:
            if identity not in full_identities_dict:
                full_identities_dict[identity] = [identities_dict[identity]]
            else:
                full_identities_dict[identity].append(identities_dict[identity])

    for identity in full_identities_dict:
        full_identities_dict[identity].sort()
        full_identities_dict[identity] = [full_identities_dict[identity][int(len(full_identities_dict[identity])*0.975)-1], 
                                          full_identities_dict[identity][int(len(full_identities_dict[identity])*0.025)-1]]
    
    return full_identities_dict

def get_max_score(score_dict):
    max_ifpr = ['', 0, 0]
    for identity in score_dict:
        if max_ifpr[0] == '':
            max_ifpr[0] = identity
            max_ifpr[1] = score_dict[identity][0]
            max_ifpr[2] = score_dict[identity][1]
        else:
            if score_dict[identity][0] > max_ifpr[1]:
                max_ifpr[0] = identity
                max_ifpr[1] = score_dict[identity][0]
                max_ifpr[2] = score_dict[identity][1]

    return max_ifpr
    
    
def identity_ifprs(dataset_path:str, label_cols:list, dataset_name:str, score_types:list, data_subset:str='', other_table=False, identity_type="big") -> dict:
    
    save_path = 'Data/ifpr_table.csv' if identity_type == 'big' else 'Data/ifpr_table_sub_identities.csv'
    other_save_path = f"Data/{dataset_name}.csv" if identity_type == 'big' else f"Data/{dataset_name}_sub_ids.csv"
    
    if len(score_types) != len(label_cols):
        print("Score and labels lists don't match")
        return
    ME_info = {'OpenAI': 'OpenAI_ME_bool', 'Llama Guard': 'OctoAI_ME_bool', 'Anthropic': 'Anthropic_ME_bool'}
    return_dict = {}
    for label_col, score_type in zip(label_cols, score_types):
        data = pd.read_csv(dataset_path)
        data.dropna(subset=label_col, inplace=True)
        if "Dixon" in dataset_path:
            data[label_col] = data[label_col].apply(lambda x: 0 if x == 'nontoxic' else 1)
        categories = get_movies_categories('big')

        max_ifpr_dict = {}
        return_dict[label_col] = {}
        for ME, me_col in zip(ME_info.keys(), ME_info.values()):
            print(ME)
            data.dropna(subset=me_col, inplace=True)
            true_labels = data[label_col].tolist()
            me_flags = data[me_col].tolist()
            overall_fpr = score_calculator({'true' : true_labels, 'pred': me_flags}, score='FPR')
            if identity_type == 'big':
                id_col = 'Big_identity'
                identities_dict = {identity.lower() : {'true' : [], 'pred' : []} for category in categories for identity in categories[category]}
            else:
                id_col = 'Sub_Identities'
                identities_dict = {}
            data_identities = data[id_col].tolist()
                    
            for identities, label, flag in zip(data_identities, true_labels, me_flags):
                if type(identities) == float:
                    continue
                identities = identities.lower().split(';')
                for identity in identities:
                    if identity not in identities_dict:
                        identities_dict[identity] = {'true' : [], 'pred' : []}
                    identities_dict[identity]['true'].append(label)
                    identities_dict[identity]['pred'].append(flag)

            identities_dict = {identity : [round(score_calculator(identities_dict[identity], score='FPR')/overall_fpr, 2),\
                confusion_matrix(identities_dict[identity]['true'], identities_dict[identity]['pred'], labels=[0,1]).ravel()[1]] \
                    for identity in identities_dict}
            bootstrap_data = bootstrap_ifprs(data, id_col, me_col, label_col)
            for bootstrap_identity in bootstrap_data:
                identities_dict[bootstrap_identity].append(bootstrap_data[bootstrap_identity])
            max_ifpr_dict[ME] = get_max_score(identities_dict)
            return_dict[label_col][ME] = identities_dict
        
        if not other_table:   
            row = {r"\textbf{Dataset}": dataset_name,
                    r"\textbf{Data subset}": data_subset,
                    r"\textbf{Score type}": score_type
                    }
            
            for ME, ifpr_data in zip(max_ifpr_dict.keys(), max_ifpr_dict.values()):
                row.update({
                    r'\textbf{ ' + ME + ' ifprscore}' : ifpr_data[1],
                    r'\textbf{ ' + ME + ' argmax}' : ifpr_data[0],
                    r'\textbf{ ' + ME + ' FP}' : ifpr_data[2],
                })
            try:
                table = pd.read_csv(save_path)
                table = pd.concat([table, pd.DataFrame(row, index=[0])])
            except Exception as e:
                table = pd.DataFrame(row, index=[0])
            table.drop_duplicates(inplace=True, subset=[r"\textbf{Dataset}", r'\textbf{Data subset}', r"\textbf{Score type}"], keep="last")
            table.to_csv(save_path, index=False)
        
    if other_table:
        df_dict = {}
        for label in return_dict:
            for ME in return_dict[label]:
                for identity in return_dict[label][ME]:
                    if identity not in df_dict:
                        df_dict[identity] = {r'\textbf{Identity group}' : identity}
                    df_dict[identity].update({
                        r'\textbf{ ' + ME + ' ifprscore}' : return_dict[label][ME][identity][0],
                        r'\textbf{ ' +  ME + ' FP}' : return_dict[label][ME][identity][1],
                        r'\textbf{ ' +  ME + ' iFPR CI}' : return_dict[label][ME][identity][2],
                    })

        try:
            table = pd.read_csv(other_save_path)
            table = pd.merge(table, pd.DataFrame(list(df_dict.values())), on=[r'\textbf{Identity group}'])

        except Exception as e:
            table = pd.DataFrame(df_dict.values())
        
        table.to_csv(other_save_path, index=False)
        
    return return_dict
    

def make_bounds():
    files = ['Data/Dixon/Data_with_ME.csv', 'Data/Markov/data_with_ME.csv', 'Data/Movies/TMDB_with_ME.csv',
             'Data/Stormfront/stormfront_data_ME.csv', 'Data/Tweets/offensive_twitter_data_with_ME.csv',
             'Data/Tweets/hate_twitter_data_with_ME.csv', 'Data/TV Shows/short_TMDB_with_ME.csv',
             'Data/TV Shows/mid_wiki_with_ME.csv', 'Data/TV Shows/long_IMDB_with_ME.csv', 'Data/CJadams/data_with_ME.csv']

    dataset = []
    for file in files:
        dataset.extend(pd.read_csv(file)['OpenAI_ME_responses'].tolist())
    responses = parse_ME_response(dataset)
    category_bounds = {}
    for response in responses:
        for category in response["categories"].keys():
            if category not in category_bounds.keys():
                category_bounds[category] = {"upper_bound" : np.inf, "lower_bound" : -np.inf}
            if response["categories"][category]:
                if float(response["category_scores"][category]) < category_bounds[category]["upper_bound"]:
                    category_bounds[category]["upper_bound"] = float(response["category_scores"][category])
            else:
                if float(response["category_scores"][category]) > category_bounds[category]["lower_bound"]:
                    category_bounds[category]["lower_bound"] = float(response["category_scores"][category])
    upper = []
    lower = []
    categories = []
    
    for category in category_bounds.keys():
        upper.append(category_bounds[category]["upper_bound"])
        lower.append(category_bounds[category]["lower_bound"])
        categories.append(category)
    
    bounds_df = pd.DataFrame()
    bounds_df["Categories"] = categories
    bounds_df["lower_bound"] = lower
    bounds_df["upper_bound"] = upper
    bounds_df.to_csv("./Data/OpenAI_flagging_bounds.csv", index=False)
    # print(bounds_df.to_latex(columns=["Categories", "lower_bound", "upper_bound"]))

    
def add_normalization():
    files = ['Data/Dixon/Data_with_ME.csv', 'Data/Markov/data_with_ME.csv', 'Data/Movies/TMDB_with_ME.csv',
             'Data/Stormfront/stormfront_data_ME.csv', 'Data/Tweets/offensive_twitter_data_with_ME.csv',
             'Data/Tweets/hate_twitter_data_with_ME.csv', 'Data/TV Shows/short_TMDB_with_ME.csv',
             'Data/TV Shows/mid_wiki_with_ME.csv', 'Data/TV Shows/long_IMDB_with_ME.csv', 'Data/CJadams/data_with_ME.csv']
    
    data = pd.read_csv("./Data/OpenAI_flagging_bounds.csv")
    bounds = data.to_dict(orient='records')
    bounds = {cat['Categories']: cat['lower_bound'] for cat in bounds}
    
    for file in files: 
        normalized_scores = []
        dataset = pd.read_csv(file)
        responses = parse_ME_response(dataset['OpenAI_ME_responses'].tolist())
        for response in responses:
            try:
                normalized_dict = {}
                for category in response['category_scores'].keys():
                    normalized_dict.update({category: float(response['category_scores'][category])/bounds[category]})
                normalized_scores.append(normalized_dict)
            except Exception as e:
                normalized_scores.append({})
        dataset['OpenAI_normalized'] = normalized_scores
        dataset.to_csv(file, index=False)
        print(f'saved {file}')


def comparison_chart_openAI():
    files = ['Data/Dixon/Data_with_ME.csv', 'Data/Markov/data_with_ME.csv', 'Data/Movies/TMDB_with_ME.csv',
             'Data/Stormfront/stormfront_data_ME.csv', 'Data/Tweets/offensive_twitter_data_with_ME.csv',
             'Data/Tweets/hate_twitter_data_with_ME.csv', 'Data/TV Shows/short_TMDB_with_ME.csv',
             'Data/TV Shows/mid_wiki_with_ME.csv', 'Data/TV Shows/long_IMDB_with_ME.csv', 'Data/CJadams/data_with_ME.csv']
    toxic_cols = ["toxicity", "Toxic", 'PG-13 score', 'label', 'Toxicity', 'Toxicity']
    
    chart_dict = {}
    all_scores = []
    for file, toxic_col in zip(files, toxic_cols):
        for score_dict, identities, toxic in pd.read_csv(file)[['OpenAI_normalized', "Big_identity", toxic_col]].to_numpy():
            if type(toxic) == str:
                toxic = 1 if toxic == 'toxic' else 0
            if type(identities) != float:
                identities = identities.split(";")
                for identity in identities:
                    if identity.lower() not in chart_dict.keys():
                        chart_dict[identity.lower()] = {1: [], 0:[]}
                    chart_dict[identity.lower()][toxic].append(max(ast.literal_eval(score_dict).values()))
            all_scores.append(max(ast.literal_eval(score_dict).values()))
    means_cols = []
    deviations = []
    
    for identity in chart_dict.keys():
        means_cols.append([identity, np.mean(chart_dict[identity][1]), np.mean(chart_dict[identity][0])])
        deviations.append([identity, np.std(chart_dict[identity][1]), np.std(chart_dict[identity][0])])

    means = pd.DataFrame(means_cols, columns=["Big_identity", 'mean_true_max', 'mean_false_max'])
    # columns are standard deviations not means, need col name to be consitent with above
    deviations = pd.DataFrame(deviations, columns=["Big_identity", 'mean_true_max', 'mean_false_max'])
    means.set_index("Big_identity", inplace=True)
    deviations.set_index("Big_identity", inplace=True)
    
    fig, ax = plt.subplots()
    means.plot.bar(yerr=deviations, ax=ax, capsize=4, rot=0, title='Mean of max normalized scores per big identity',
        figsize=(25, 20),
        xlabel='Big Identities',
        ylabel='Mean of max normalized scores',
        width=0.8,
        fontsize=15,)
    plt.ylim(0)
    plt.axhline(y = np.mean(all_scores), color = 'black', linestyle = '-', label=f"Average max normalized score")
    plt.savefig("./plots/OpenAI_comparison_chart.png", pad_inches=1, bbox_inches='tight')


def comparison_chart_other(ME):
    files = ['Data/Dixon/Data_with_ME.csv', 'Data/Markov/data_with_ME.csv', 'Data/Movies/TMDB_with_ME.csv',
             'Data/Stormfront/stormfront_data_ME.csv', 'Data/Tweets/offensive_twitter_data_with_ME.csv',
             'Data/Tweets/hate_twitter_data_with_ME.csv', 'Data/TV Shows/short_TMDB_with_ME.csv',
             'Data/TV Shows/mid_wiki_with_ME.csv', 'Data/TV Shows/long_IMDB_with_ME.csv', 'Data/CJadams/data_with_ME.csv']
    toxic_cols = ["toxicity", "Toxic", 'PG-13 score', 'label', 'Toxicity', 'Toxicity']
    chart_dict = {}
    all_scores = []
    
    for file, toxic_col in zip(files, toxic_cols):
        data = pd.read_csv(file)
        try:
            if ME == "PerspectiveAI":
                ME_col = "perspective_ME_responses"
                data[ME_col] = data[ME_col].apply(lambda x: {"Toxic": ast.literal_eval(x)\
                    ["attributeScores"]["TOXICITY"]["spanScores"][0]["score"]["value"]}\
                        if "API error" not in x and "languageNotSupportedByAttributeError" not in x else {})
            elif ME == "Google":
                ME_col = "Google_ME_responses"
                data[ME_col] = data[ME_col].apply(lambda x: ast.literal_eval(x)\
                    if "API error" not in x else {})
            else:
                continue
        except KeyError as ke:
            continue
        
        for score_dict, identities, toxic in data[[ME_col, "Big_identity", toxic_col]].to_numpy():
            if type(toxic) == str:
                toxic = 1 if toxic == 'toxic' else 0
            if type(identities) != float:
                identities = identities.split(";")
                for identity in identities:
                    if identity.lower() not in chart_dict.keys():
                        chart_dict[identity.lower()] = {1: [], 0:[]}
                    chart_dict[identity.lower()][toxic].append(max(score_dict.values())) if len(score_dict) > 0 else None
            all_scores.append(max(score_dict.values())) if len(score_dict) > 0 else None
        
    means_cols = []
    deviations = []
    
    for identity in chart_dict.keys():
        means_cols.append([identity, np.mean(chart_dict[identity][1]), np.mean(chart_dict[identity][0])])
        deviations.append([identity, np.std(chart_dict[identity][1]), np.std(chart_dict[identity][0])])

    means = pd.DataFrame(means_cols, columns=["Big_identity", 'mean_true_max', 'mean_false_max'])
    # columns are standard deviations not means, need col name to be consitent with above
    deviations = pd.DataFrame(deviations, columns=["Big_identity", 'mean_true_max', 'mean_false_max'])
    means.set_index("Big_identity", inplace=True)
    deviations.set_index("Big_identity", inplace=True)
    
    fig, ax = plt.subplots()
    means.plot.bar(yerr=deviations, ax=ax, capsize=4, rot=0, title=f'Mean of max scores per big identity {ME}',
        figsize=(25, 20),
        xlabel='Big Identities',
        ylabel='Mean of max scores',
        width=0.8,
        fontsize=15,)
    plt.ylim(0)
    plt.axhline(y = np.mean(all_scores), color = 'black', linestyle = '-', label=f"Average max score")
    plt.savefig(f"./plots/{ME}_comparison_chart.png", pad_inches=1, bbox_inches='tight')

   
def dataset_table(append=True):
    if append:
        dataset_table_helper("Data/CJadams/data.csv", "Jigsaw Kaggle (cjadams et al. 2017)", "comment_text", ["-"])
        dataset_table_helper("Data/Dixon/Data_with_ME.csv", "Jigsaw Bias (Dixon et al. 2018)", "phrase", ["-"])
        dataset_table_helper("Data/Stormfront/stormfront_data_ME.csv", "Stormfront (De Gibert et al. 2018)", "Sentence", ["-"])
        
        dataset_table_helper("Data/Tweets/hate_twitter_data_with_ME.csv", "TweetEval (Barbieri et al. 2020)", "Text", ["Hate"])
        dataset_table_helper("Data/Tweets/offensive_twitter_data_with_ME.csv", "TweetEval (Barbieri et al. 2020)", "Text", ["Offensive"])
        dataset_table_helper("Data/Markov/data_with_ME.csv", "OpenAI (Markov et al. 2023)", "prompt", ['-'])
        dataset_table_helper("Data/Movies/TMDB_with_ME.csv", "Movie Plots", "plots", ["-"])
        dataset_table_helper("Data/TV Shows/long_IMDB_with_ME.csv", "TV Synopses", "synopsis_with_character_names", ["Long IMDB"])
        dataset_table_helper("Data/TV Shows/mid_wiki_with_ME.csv", "TV Synopses", "wiki_descs", ["Medium Wiki"])
        dataset_table_helper("Data/TV Shows/short_TMDB_with_ME.csv", "TV Synopses", "episode-overview", ["Short TMDB"])
        dataset_table_helper("Data/Combined/genAI_combined.csv", "GenAI", "text", ["-"])
        dataset_table_helper("Data/Combined/traditional_combined.csv", "Traditional", "text", ["-"])
    else:
        print(pd.read_csv("./Data/datasets_table.csv").astype('str').to_latex(index=False), '\n')
        print(pd.read_csv("./Data/datasets_table_identity.csv").astype('str').to_latex(index=False))
        

def dataset_table_helper(dataset_path, dataset_name, column, data_subsets):
    df = pd.read_csv(dataset_path) if ".csv" in dataset_path else pd.read_json(dataset_path,  lines=True)
    mean = np.mean(df[column].apply(lambda x: len(x.split()) if type(x) != float else 0))
    
    row = {r"\textbf{Dataset}": dataset_name,
            r"\textbf{Data subset}": ":".join(data_subsets),
            r"\textbf{Total (items)}": df.shape[0],
            r"\textbf{Avg Word Length}" : round(mean, 2),
            }
    
    identity_row = {
            r"\textbf{Dataset}": dataset_name,
            r"\textbf{Data subset}": ":".join(data_subsets),
    }
    
    #TODO: Add combined
    stats = calculate_id_stats(df)
    identity_row.update(stats)
    
    try:
        table = pd.read_csv("./Data/datasets_table.csv")
        table = pd.concat([table, pd.DataFrame(row, index=[0])])
    except Exception as e:
        table = pd.DataFrame(row, index=[0])
    
    table.drop_duplicates(inplace=True, subset=[r"\textbf{Dataset}", r'\textbf{Data subset}'], keep="last")
    table.to_csv("./Data/datasets_table.csv", index=False)
    
    try:
        table = pd.read_csv("./Data/datasets_table_identity.csv")
        table = pd.concat([table, pd.DataFrame(identity_row, index=[0])])
    except Exception as e:
        table = pd.DataFrame(identity_row, index=[0])
    table.drop_duplicates(inplace=True, subset=[r"\textbf{Dataset}", r'\textbf{Data subset}'], keep="last")
    table.to_csv("./Data/datasets_table_identity.csv", index=False)
    

def calculate_id_stats(data):
    big_ids = data["Big_identity"].tolist()

    identity_ct = {
                    r'\textbf{Non-white}': 0,
                    r'\textbf{White}': 0,
                    r'\textbf{Men}': 0,
                    r'\textbf{Women}': 0,
                    r'\textbf{Christian}': 0,
                    r'\textbf{Non-christian}': 0,
                    r'\textbf{LGBT}': 0,
                    r'\textbf{Straight}': 0,
                    r'\textbf{Disability}': 0,
            }
    
    for ids in big_ids:
        sub_ids = ids.split(";") if type(ids) == str else []
        for subs in sub_ids:
            identity = r'\textbf{' + subs.capitalize() + '}'
            if identity in identity_ct.keys():
                identity_ct[identity] += 1
            if "lgbt" in subs.lower():
                identity_ct[r"\textbf{LGBT}"] += 1
        
    return identity_ct


def update_overall_table(dataset_name: str, ME_name: str, category: str, identity:str,
                         fpr_score: float, overall_score:float, all_fpr: str):
    data = {"Dataset": dataset_name,
            "ME": ME_name,
            "Category": category,
            "Identity" : identity,
            "Benchmark score": fpr_score,
            "Dataset score": overall_score,
            "Other scores": all_fpr}
    try:
        table = pd.read_csv("./Data/analysis_table.csv")
        table = pd.concat([table, pd.DataFrame(data, index=[0])])
    except Exception as e:
        table = pd.DataFrame(data, index=[0])
    
    table.drop_duplicates(inplace=True, subset=["Dataset", "ME", "Category"], keep="last")
    table.to_csv("./Data/analysis_table.csv", index=False)


def conv_openAI_ME_data(ME_responses) -> None:
    """Converts ME responses to bool values and makes new csv"""
    conv_ME_output = [1 if ME_response["results"][0]["flagged"] else 0 for ME_response in ME_responses]
    
    return conv_ME_output


def parse_ME_response(ME_data:list) -> list:
    """Converts each response from string to dictionary then returns the results"""
    return [ast.literal_eval(ME_dict)["results"][0] for ME_dict in ME_data]


def get_splits(string:str):

    string = string.split("(")[0].replace("or", ",").replace("/", ",").replace("and", ",")
    terms = string.split(",")
    
    return [term.strip() for term in terms if len(term) > 2]
    

def get_wiki_identities(page_title):
    """
    gets movies list from pages with tables

    Args:
        page_title (str): name of page

    Returns:
        list: all movies found
    """
    try:
        identity_slurs_dict = {}
        page = wiki.page(page_title, auto_suggest=False).html().encode("UTF-8")
        dfs = pd.read_html(page)
        for df in dfs:
            if "Targets" in df.columns and "Term" in df.columns:
                for targets, terms in zip(df["Targets"].tolist(), df["Term"].tolist()):
                    if type(targets) == str:
                        targets = targets.split(",")  
                        terms = get_splits(terms)
                        for target in targets:
                            if target in identity_slurs_dict.keys():
                                identity_slurs_dict[target].extend(terms)
                            else:
                                identity_slurs_dict[target] = terms
                    else:
                        print(terms)
    except ValueError as ve:
        print("Error:", ve)
    
    return filter_words(identity_slurs_dict)


def get_lgbt_wiki():
    terms = titles_recursively(category='Category:LGBT-related slurs', depth=2)
    terms = [term.split("(")[0].strip() for term in terms]
    file = pd.read_csv('Data/wiki_identity.csv')
    file.loc[len(file.index)] = ["LGBT-related", "LGBT-related", list(set(terms))]
    file.to_csv('Data/wiki_identity.csv', index=False)


def filter_words(identity_dict):
    lesbian = ['lgbt-related', 'lesbian']
    gay = ['lgbt-related', 'gay']
    bisexual = ['lgbt-related', 'bisexual']
    homosexual = ['lgbt-related', 'homosexual']
    male = ['men', 'male', ' man ', ' men ', ' man.', ' men.', ' man,', ' men,']
    female = ['women', 'female', 'woman', 'women']
    middle_eastern = ['Race-Ethnicity', 'middle eastern']
    muslim = ['religon', 'muslim']
    jewish = ['religon', 'jewish']
    buddhist = ['religon', 'buddhist']
    sikh = ['religon', 'sikh']
    taoist = ['religon', 'taoist']
    middle_aged = ['middle aged']
    trans = ['lgbt-related', 'trans', 'transgender', 'nonbinary']
    lgbtq = ['lgbt-related', 'lgbt', 'lgbtq', 'queer']
    straight = ['straight', 'heterosexual']
    black = ['Race-Ethnicity', 'black', 'african', 'african american']
    white = ['Race-Ethnicity', 'white', 'european']
    latinx = ['Race-Ethnicity', 'latinx', 'latina', 'latino', 'hispanic', 'mexican']
    asian = ['Race-Ethnicity', 'asian', 'indian', 'chinese', 'japanese']
    christian = ['religon', 'christian', 'catholic', 'protestant']
    disabled = ['disabled', 'blind', 'deaf', 'paralyzed']

    identities = [lesbian,gay,bisexual,homosexual,male,female,
                middle_eastern,muslim,jewish,buddhist,sikh,
                taoist,middle_aged,trans,lgbtq,straight,black,white,latinx,
                asian,christian,disabled]

    id_dict = {}

    all_ids = []

    for identity in identities:
        id_dict.update({key: identity[0] for key in identity})
        for id in identity:
            all_ids.append(id)
    
    final_dict = {}
    hold_dict = {}
    for word in all_ids:
        for key in identity_dict.keys():
            if word.lower() in key.lower():
                if (id_dict[word], word) in final_dict.keys(): 
                    for term in identity_dict[key]:
                        final_dict[(id_dict[word], word)].append(term) if term not in final_dict[(id_dict[word], word)] else None
                else:
                    final_dict[(id_dict[word], word)] = list(set(identity_dict[key]))
                if word in hold_dict.keys():
                    hold_dict[word].extend(identity_dict[key])
                else:
                    hold_dict[word] = identity_dict[key]
    
    mappings = pd.DataFrame(columns=["Identity category", "Sub_identity", "Terms"])
    for key, value in final_dict.items():
        row = {
            "Identity category": key[0],
            "Sub_identity": key[1],
            "Terms": str(value)
        }
        mappings = pd.concat([mappings, pd.DataFrame(row, index=[0])])
        
    mappings.to_csv("Data/wiki_identity.csv", index=False)


def score_calculator(metric_dict:dict, score=None) -> dict:
    cm = confusion_matrix(metric_dict["true"], metric_dict["pred"], labels=[0,1])
    scores_dict = {"Accuracy": 0, "Precision": 0, "Recall": 0}

    scores_dict["Accuracy"] = round(accuracy_score(metric_dict["true"], metric_dict["pred"]), 4)
    scores_dict["Precision"] = round(precision_score(metric_dict["true"], metric_dict["pred"], zero_division=0), 4)
    scores_dict["Recall"] = round(recall_score(metric_dict["true"], metric_dict["pred"], zero_division=0), 4)
    scores_dict["TPR"] = round(cm[1,1] / (cm[1,1] + cm[0,1]) if cm[1,1] + cm[0,1] > 0 else 1, 4)
    scores_dict["FPR"] = round(cm[0, 1] / (cm[0, 0] + cm[0, 1]) if cm[0, 0] + cm[0, 1] > 0 else 0, 4)
      
    return scores_dict[score] if score is not None else scores_dict


def get_terms_mapping()->dict:
    """
    compiles the list of terms we need for manual tagging
    Returns:
        dict: dictionary containing a list of all terms with their big and individual identities
    """
    wiki_file = pd.read_csv("Data/wiki_identity.csv")
    term_dict = {}
    for category, identity, terms in zip(wiki_file["Identity category"], wiki_file["Sub_identity"], wiki_file["Terms"]):
        terms = ast.literal_eval(terms)
        for term in terms:
            term_dict[term] = [category, identity]
            
    neutral_list = pd.read_csv("Data/neutral_identity_terms.csv")
    for category, identity, terms in zip(neutral_list["Identity category"], neutral_list["Sub_identity"], neutral_list["Terms"]):
        terms = ast.literal_eval(terms)
        for term in terms:
            term_dict[term] = [category, identity]

    return term_dict


def get_dixon_categories():
    dixon_data = pd.read_csv("Data/Dixon/Data_with_identities.csv")
    caterories_dict = {}
    
    for group, identity in zip(dixon_data["Big_identity"].tolist(), dixon_data["Sub_Identities"].tolist()):
        if group in caterories_dict:
            caterories_dict[group].append(identity) if identity not in caterories_dict[group] else None
        else:
            caterories_dict[group] = [identity]
    
    return caterories_dict


def get_movies_categories(identity_type) -> dict:
    tagged_files = os.listdir("./Data/Movies/tags")
    sub_categories_dict = {}
    big_categories_dict = {}
    for category in tagged_files:
        identitiy_bins = os.listdir(f"./Data/Movies/tags/{category}")
        sub_categories_dict[category] = []
        big_categories_dict[category] = []
        for identity_bin in identitiy_bins:
            bin_files = os.listdir(f"./Data/Movies/tags/{category}/{identity_bin}")
            big_categories_dict[category].append(identity_bin)
            for file in bin_files:
                sub_categories_dict[category].append(file.split("_tagged_movies.csv")[0])
                
    return sub_categories_dict if identity_type == "Sub_Identities" else big_categories_dict


def ME_score_analysis(identity_type, data_type, file, ME, ex=""):
    plt.figure(figsize=(15, 10))
    data = pd.read_csv(f"Data/{data_type}/{file}")
    if ME == "PerspectiveAI":
        ME_col = "perspective_ME_responses"
        data[ME_col] = data[ME_col].apply(lambda x: ast.literal_eval(x)\
            ["attributeScores"]["TOXICITY"]["spanScores"][0]["score"]["value"]\
                if "API error" not in x and "languageNotSupportedByAttributeError" not in x else "Error")
    elif ME == "Google":
        ME_col = "Google_ME_responses"
        data[ME_col] = data[ME_col].apply(lambda x: ast.literal_eval(x)['Toxic']\
            if "API error" not in x else "Error")
    else:
        raise("Invalid ME")
    
    data = data[data[ME_col] != "Error"]
    
    overall_avg = np.mean(data[ME_col])
    overall_std = np.std(data[ME_col])
    identity_rating_scores = {}
    identites_dict = {"big" : "Big_identity", "small": "Sub_Identities"}

    identity_col = identites_dict[identity_type]
    for score, identities in zip(data[ME_col], data[identity_col]):
        sub_ids = identities.split(";") if type(identities) != float else ["No identity"]     
        for ids in sub_ids:
            if ids.lower() in identity_rating_scores.keys():
                identity_rating_scores[ids.lower()].append(score)
            elif ids:
                identity_rating_scores[ids.lower()] = [score]

    means = []
    deviations = []
    for key in identity_rating_scores.keys():
        mean, std = np.mean(identity_rating_scores[key]), np.std(identity_rating_scores[key])
        identity_rating_scores[key] = {"Average": mean, "Standard deviation": std}
        means.append(mean)
        deviations.append(std)
        
    if not os.path.exists(f"./Data/{data_type}/{ME}"):
        os.makedirs(f"./Data/{data_type}/{ME}")
        
    df = pd.DataFrame(columns=["Identity", 'Average score'])
    df["Identity"] = identity_rating_scores.keys()
    df["Average score"] = means
    df["Standard Deviation"] = deviations
    df.loc[len(df.index)] = ["All", overall_avg, overall_std]
    df.sort_values(by=["Average score"], inplace=True, ascending=False)
    df.to_csv(f"./Data/{data_type}/{ME}/{ex}{identity_type}_rating_scores.csv", index=False)
    
    cat_type = "Big_identity" if identity_type == "big" else "Sub_Identities"
    if identity_type == "big":
        categories = get_movies_categories(cat_type)
    elif data_type == "Movies":
        categories = get_movies_categories(cat_type)
    elif data_type == "Dixon" or data_type == "Tweets" or data_type == "Markov"\
         or data_type == "Stormfront":
        categories = get_dixon_categories()
    
    for category in categories.keys():
        identites = categories[category]
        metrics = []
        std = []
        for identity in identites:
            identity = identity.lower()
            metrics.append(identity_rating_scores[identity]["Average"])\
                if identity in identity_rating_scores.keys() else metrics.append(0)
            std.append(identity_rating_scores[identity]["Standard deviation"])\
                if identity in identity_rating_scores.keys() else std.append(0)
        plt.bar(identites, metrics, yerr=std, align='center', width=0.7, label=category)
    
    plt.xticks(rotation=90)
    plt.axhline(y = overall_avg, color = 'black', linestyle = '-', label=f"Average score")
    plt.xlabel('Identities')
    plt.ylabel("Average Score")
    plt.ylim(0)
    plt.title(f'Average Scores by {identity_type} identities ({ME})')
    save_path = f"./plots/{data_type}/{ME}/{ex}{identity_type}_identities_chart.png"
    
    if not os.path.exists(f"./plots/{data_type}/{ME}/"):
        os.makedirs(f"./plots/{data_type}/{ME}/")
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.8)
    plt.clf()
    plt.close()


def openAI_analysis(identity_type, data_type, file, toxic_col, ex="",ME="OpenAI"):
       
    data = pd.read_csv(f"Data/{data_type}/{file}")
    if data_type == "Dixon":
        data[toxic_col] = data[toxic_col].apply(lambda x: 0 if x == 'nontoxic' else 1,)
    identity_dict = {}
    dataset_score = score_calculator({"true": data[toxic_col], "pred": data[f"{ME}_ME_bool"]})
    for toxic, identities, ME_score in zip(data[toxic_col], data[identity_type], data[f"{ME}_ME_bool"]):
        sub_ids = identities.split(";") if type(identities) != float else ["No identity"]
        for identity in sub_ids:
            if identity not in identity_dict.keys():
                identity_dict[identity] = {"true": [toxic], "pred": [ME_score]}
            else:
                identity_dict[identity]["true"].append(toxic)
                identity_dict[identity]["pred"].append(ME_score)
    
    identities_fpr = {}
    max_fpr = {"identity": "", "FPR": -np.inf}
    for identity in identity_dict.keys():
        identity_dict[identity] = score_calculator(identity_dict[identity])
        identities_fpr[identity] = identity_dict[identity]["FPR"]
        if identity_dict[identity]["FPR"] > max_fpr["FPR"]:
            max_fpr["FPR"] = identity_dict[identity]["FPR"]
            max_fpr["identity"] = identity
                
    graph_rating_acc(ME, data_type, identity_type, "FPR", identity_dict, dataset_score, ex=ex)
    graph_rating_acc(ME, data_type, identity_type, "Accuracy", identity_dict, dataset_score, ex=ex)
    
    update_overall_table(data_type, ME, f"FPR", \
        max_fpr['identity'], max_fpr['FPR'], dataset_score["FPR"], str(identities_fpr))
    
    if not os.path.exists(f"./Data/{data_type}/{ME}"):
        os.makedirs(f"./Data/{data_type}/{ME}/")
    
    identity_type = "Small_identities" if identity_type.lower() == "sub_identities" else "Big_identities"
    df = pd.DataFrame(columns=["Identity", "Stats"])
    for key in identity_dict.keys():
        df.loc[len(df.index)] = [key, identity_dict[key]]
    df.loc[len(df.index)] = ["All", dataset_score]
    df.to_csv(f"./Data/{data_type}/{ME}/{ex}{identity_type}_rating_scores.csv", index=False)
    

def graph_rating_acc(ME, data_type, identity_type, metric:str, identities_dict:dict, category_scores:dict, ex="") -> None:

    identity_path = "Small identities" if identity_type.lower() == "sub_identities" else "Big identities"
    if not os.path.exists(f"./plots/{data_type}/{ME}/{identity_path}/{metric}"):
        os.makedirs(f"./plots/{data_type}/{ME}/{identity_path}/{metric}")
    
    if 'Big' in identity_type:
        categories_dict = get_movies_categories(identity_type)
    elif data_type == "Movies":
        categories_dict = get_movies_categories(identity_type)
    elif data_type == "Dixon" or data_type == "Tweets" or data_type == "Markov" or data_type == "Stormfront":
        categories_dict = get_dixon_categories()
    
    plt.figure(figsize=(15, 10))
    plt.xticks(rotation=90)

    for category in categories_dict.keys():
        identities = categories_dict[category]
        new_ids = []
        metrics = []
        for identity in identities:
            identity = identity.lower()
            if identity in identities_dict.keys():
                metrics.append(identities_dict[identity][metric])
                new_ids.append(identity.capitalize())
        plt.bar(new_ids, metrics, width=0.7, label=category)
        
    plt.axhline(y = category_scores[metric], color = 'black', linestyle = '-', label=f"Average {metric}")
    plt.xlabel('Identities')
    plt.ylabel(metric)
    plt.title(f'Flagged {metric} by identity')
    save_path = f"./plots/{data_type}/{ME}/{identity_path}/{metric}/{ex}{metric.lower()}_chart.png"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.8)
    plt.clf()
    plt.close()    


def combine_data():
    if not os.path.exists('Data/Combined'):
        os.makedirs('Data/Combined')
    
    movies = pd.read_csv("Data/Movies/TMDB_with_ME.csv")[['Title', 'PG score', 'PG-13 score', 
            'rating', 'genres', 'Release year', 'plots', 'Big_identity', 'Sub_Identities', 'perspective_ME_responses',
            'OpenAI_ME_responses', 'OpenAI_ME_bool', 'OpenAI_normalized',
            'Anthropic_ME_responses', 'Anthropic_ME_bool', 'Google_ME_responses',
            'OctoAI_ME_responses', 'OctoAI_ME_bool']]
    movies.rename(columns={'plots': 'text', 'rating': 'age_rating'}, inplace=True)
    movies['true_label'] = movies['PG-13 score']
    movies['dataset_name'] = ['Movies'] * movies.shape[0]
    
    markov = pd.read_csv("Data/Markov/data_with_ME.csv")
    markov.rename(columns={'prompt': 'text'}, inplace=True)
    markov['true_label'] = markov['Toxic']
    markov['dataset_name'] = ['OpenAI'] * markov.shape[0]
    
    tv_cols = ['text', 'show_name', 'episode_title', 'Big_identity', 
            'Sub_Identities', 'age_rating', 'genres', 'PG score', 'PG-13 score', 'perspective_ME_responses',
            'OpenAI_ME_responses', 'OpenAI_ME_bool', 'OpenAI_normalized',
            'Anthropic_ME_responses', 'Anthropic_ME_bool', 'Google_ME_responses',
            'OctoAI_ME_responses', 'OctoAI_ME_bool']

    short_tv = pd.read_csv('Data/TV Shows/short_TMDB_with_ME.csv').rename(columns={'episode-overview': 'text'})[tv_cols]
    short_tv['true_label'] = short_tv['PG-13 score']
    short_tv['dataset_name'] = ['TV Shows'] * short_tv.shape[0]
    short_tv['subset_name'] = ['Short TMDB'] * short_tv.shape[0]
    
    mid_tv = pd.read_csv('Data/TV Shows/mid_wiki_with_ME.csv').rename(columns={'wiki_descs': 'text'})[tv_cols]
    mid_tv['true_label'] = mid_tv['PG-13 score']
    mid_tv['dataset_name'] = ['TV Shows'] * mid_tv.shape[0]
    mid_tv['subset_name'] = ['Medium Wiki'] * mid_tv.shape[0]
    
    long_tv = pd.read_csv('Data/TV Shows/long_IMDB_with_ME.csv').rename(columns={'synopsis_with_character_names': 'text'})[tv_cols]
    long_tv['true_label'] = long_tv['PG-13 score']
    long_tv['dataset_name'] = ['TV Shows'] * long_tv.shape[0]
    long_tv['subset_name'] = ['Long IMDB'] * long_tv.shape[0]
    
    genAI = [movies, short_tv, mid_tv, long_tv, markov]
    genAI_combined = pd.concat(genAI, ignore_index=True)#.reset_index(drop=True)
    genAI_combined['word_length'] = genAI_combined['text'].apply(lambda x: len(x.split()))
    print(genAI_combined.shape)
    print(genAI_combined.columns)
    
    genAI_combined.to_csv('Data/Combined/genAI_combined.csv', index=False)
    
    
    dixon = pd.read_csv("Data/Dixon/Data_with_ME.csv").rename(columns={'phrase': 'text'}).drop(columns=['Unnamed: 0'])
    dixon['true_label'] = dixon['toxicity'].apply(lambda x: 0 if x == 'nontoxic' else 1)
    dixon['dataset_name'] = ['Jigsaw Bias'] * dixon.shape[0]
    
    tweet1 = pd.read_csv("Data/Tweets/hate_twitter_data_with_ME.csv").rename(columns={'Text': 'text'})
    tweet1['true_label'] = tweet1['Toxicity']
    tweet1.rename(columns={'Toxicity': 'Hate'}, inplace=True)
    tweet1['dataset_name'] = ['TweetEval'] * tweet1.shape[0]
    tweet1['subset_name'] = ['Hate'] * tweet1.shape[0]
    
    tweet2 = pd.read_csv("Data/Tweets/offensive_twitter_data_with_ME.csv").rename(columns={'Text': 'text'})
    tweet2['true_label'] = tweet2['Toxicity']
    tweet2.rename(columns={'Toxicity': 'Offensive'}, inplace=True)
    tweet2['dataset_name'] = ['TweetEval'] * tweet2.shape[0]
    tweet2['subset_name'] = ['Offensive'] * tweet2.shape[0]
    
    stormfront = pd.read_csv("Data/Stormfront/stormfront_data_ME.csv").rename(columns={'Sentence': 'text', 'label': 'true_label'})
    stormfront['dataset_name'] = ['Stormfront'] * stormfront.shape[0]
    
    adams = pd.read_csv("Data/CJadams/data_with_ME.csv").rename(columns={'comment_text': 'text', 'toxicity': 'Toxicity'})
    adams['true_label'] = adams['Toxicity']
    adams['dataset_name'] = ['Jigsaw Kaggle'] * adams.shape[0]
    
    traditional = [dixon, tweet1, tweet2, stormfront, adams]
    
    traditional_combined = pd.concat(traditional, ignore_index=True)
    traditional_combined['word_length'] = traditional_combined['text'].apply(lambda x: len(x.split()))
    traditional_combined.drop(columns=['index'])
    print(traditional_combined.shape)
    print(traditional_combined.columns)
    
    traditional_combined.to_csv('Data/Combined/traditional_combined.csv', index=False)
    
    row_sum = 0
    for data in traditional:
        if len(set(list(data.columns))) != len(list(data.columns)):
            print(data.columns)
             
        row_sum += data.shape[0]
    print(row_sum)
    
    add_identity_cols_combined_data()


def add_metrics(metric_func, identity='big')->None:
    openai_labels = 'Toxic,S,H,V,HR,SH,S3,H2,V2'.split(',')
    openai_scores = ['overall flag', 'sexual', 'hate', 'violence', 'harassment', 'self-harm', 'sex./minors', 'hate/threat.', 'viol./graphic']
    adams_labels = 'toxicity,severe_toxicity,obscene,sexual_explicit,identity_attack,insult,threat'.split(',')
    adams_scores = ['overall toxicity', 'severe_toxicity', 'obscene', 'sexual_explicit', 'identity_attack', 'insult', 'threat']
    
    metric_func('Data/CJadams/data_with_ME.csv', label_cols=adams_labels, dataset_name='Jigsaw kaggle', score_types=adams_scores, identity_type=identity)
    metric_func('Data/Dixon/Data_with_ME.csv', label_cols=['toxicity'], dataset_name='Jigsaw Bias', score_types=['toxicity'], identity_type=identity)
    metric_func('Data/Stormfront/stormfront_data_ME.csv', label_cols=['label'], dataset_name='Stormfront', score_types=['hate'], identity_type=identity)
    metric_func('Data/Tweets/hate_twitter_data_with_ME.csv', label_cols=['Toxicity'], dataset_name='TweetEval', score_types=['hate'], data_subset='hate', identity_type=identity)
    metric_func('Data/Tweets/offensive_twitter_data_with_ME.csv', label_cols=['Toxicity'], dataset_name='TweetEval', score_types=['offensive'], data_subset='offensive', identity_type=identity)
    metric_func('Data/Markov/data_with_ME.csv', label_cols=openai_labels, dataset_name='OpenAI', score_types=openai_scores, identity_type=identity)
    metric_func('Data/Movies/TMDB_with_ME.csv', label_cols=['PG-13 score'], dataset_name='Movie Plots', score_types=['PG-13 appro.'], identity_type=identity)
    metric_func('Data/Movies/TMDB_with_ME.csv', label_cols=['PG score'], dataset_name='Movie Plots', score_types=['PG appro.'], identity_type=identity)
    metric_func('Data/TV Shows/short_TMDB_with_ME.csv', label_cols=['PG score', 'PG-13 score'], dataset_name='TV Synops.', score_types=['PG appro.', 'PG-13 appro.'], data_subset='Short TMDB', identity_type=identity)
    metric_func('Data/TV Shows/mid_wiki_with_ME.csv', label_cols=['PG score', 'PG-13 score'], dataset_name='TV Synops.', score_types=['PG appro.', 'PG-13 appro.'], data_subset='med. Wiki.', identity_type=identity)
    metric_func('Data/TV Shows/long_IMDB_with_ME.csv', label_cols=['PG score', 'PG-13 score'], dataset_name='TV Synops.', score_types=['PG appro.', 'PG-13 appro.'], data_subset='long IMDB', identity_type=identity)
    metric_func('Data/Combined/genAI_combined.csv', label_cols=['true_label'], dataset_name='GenAI', score_types=['True label'], identity_type=identity)
    metric_func('Data/Combined/traditional_combined.csv', label_cols=['true_label'], dataset_name='Traditional', score_types=['True label'], identity_type=identity)


def make_ME_med_charts():
    data_dict = {"Traditional": pd.read_csv('Data/Traditional.csv'), "Generative": pd.read_csv('Data/GenAI.csv')}
    ME_cols = {'Google': [r'\textbf{ Google imedianscore}', r'\textbf{ Google imedian CI}'],
               'Jigsaw': [r'\textbf{ Perspective imedianscore}', r'\textbf{ Perspective imedian CI}'],
               'OpenAI': [r'\textbf{ OpenAI imedianscore}', r'\textbf{ OpenAI imedian CI}']}
    colors = ['#1f78b4', '#33a02c', '#33a02c', '#8da0cb', '#8da0cb', '#fc8d62', '#fc8d62', '#e78ac3', '#e78ac3']
    
    for ME in ME_cols:
        medians = pd.DataFrame()
        confidence_interval = pd.DataFrame()
        for data in data_dict:
            medians['Identity'] = data_dict[data][r'\textbf{Identity group}']
            confidence_interval['Identity'] = data_dict[data][r'\textbf{Identity group}']
            medians[data] = data_dict[data][ME_cols[ME][0]]
            confidence_interval[data] = data_dict[data][ME_cols[ME][1]].apply(lambda x: np.array(ast.literal_eval(x)))

        medians.set_index('Identity', inplace=True)
        confidence_interval.set_index('Identity', inplace=True)
        fig, ax = plt.subplots()
        
        # Calculate errors as a list of arrays
        errors = [medians['Traditional'] - confidence_interval['Traditional'].apply(lambda x: x[1]),
                  confidence_interval['Traditional'].apply(lambda x: x[0]) - medians['Traditional']]
        
        medians['Traditional'].plot.bar(legend=False, yerr=errors, ax=ax, capsize=4, title=f'{ME}', rot=45,
                                        label='Traditional', figsize=(25, 20), xlabel='Identity groups', ylabel='Speech suppression',
                                        color=colors, width=-0.4, align='edge', hatch='/')
        
        errors = [medians['Generative'] - confidence_interval['Generative'].apply(lambda x: x[1]),
                  confidence_interval['Generative'].apply(lambda x: x[0]) - medians['Generative']]
        
        medians['Generative'].plot.bar(legend=True, yerr=errors, ax=ax, capsize=4,
                                       label='Generative', figsize=(25, 20), color=colors, width=0.4, align='edge', hatch='*')
        
        traditional_patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='/', label='Traditional')
        generative_patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='*', label='Generative')
        plt.legend(handles=[traditional_patch, generative_patch])
        plt.ylim(0)
        plt.axhline(y=1.0, color='black', linestyle='-')
        plt.xticks(rotation=45)
        plt.savefig(f"./plots/{ME}Imedians.png", pad_inches=1, bbox_inches='tight')
        plt.clf()
        
        
def make_ME_fpr_charts():
    data_dict = {"Traditional" : pd.read_csv('Data/Traditional.csv'), "Generative": pd.read_csv('Data/GenAI.csv')}
    ME_cols = {'Anthropic': [r'\textbf{ Anthropic ifprscore}', r'\textbf{ Anthropic iFPR CI}'],
            'Llama Guard': [r'\textbf{ Llama Guard ifprscore}', r'\textbf{ Llama Guard iFPR CI}'],
            'OpenAI': [r'\textbf{ OpenAI ifprscore}', r'\textbf{ OpenAI iFPR CI}']
            }
    colors = ['#1f78b4', '#33a02c', '#33a02c', '#8da0cb', '#8da0cb', '#fc8d62', '#fc8d62', '#e78ac3', '#e78ac3']
    
    for ME in ME_cols:
        ifprs = pd.DataFrame()
        confidence_interval = pd.DataFrame()
        for data in data_dict:
            if 'Identity' not in ifprs.columns.tolist():
                ifprs['Identity'] = data_dict[data][r'\textbf{Identity group}']
                confidence_interval['Identity'] = data_dict[data][r'\textbf{Identity group}']
            else:
                assert(ifprs['Identity'].tolist() == data_dict[data][r'\textbf{Identity group}'].tolist())

            ifprs[data] = data_dict[data][ME_cols[ME][0]]
            confidence_interval[data] = data_dict[data][ME_cols[ME][1]].apply(lambda x: np.array(ast.literal_eval(x)))
            
            
        ifprs.set_index('Identity', inplace=True)
        confidence_interval.set_index('Identity', inplace=True)
        errors = [ifprs['Traditional'] - confidence_interval['Traditional'].apply(lambda x: x[1]),
                  confidence_interval['Traditional'].apply(lambda x: x[0]) - ifprs['Traditional']]
        ifprs['Traditional'].plot.bar(legend=False, title=f'{ME}',yerr=errors,
                                      label='Traditional', figsize=(25, 20), rot=45,
                xlabel='Identity groups', ylabel='Speech suppression',color=colors, width=-0.4, align='edge', hatch='/')
        
        errors = [ifprs['Generative'] - confidence_interval['Generative'].apply(lambda x: x[1]),
                  confidence_interval['Generative'].apply(lambda x: x[0]) - ifprs['Generative']]
        ifprs['Generative'].plot.bar(legend=True, label='Generative', figsize=(25, 20), yerr=errors,
        xlabel='Identity groups',
        ylabel='Speech suppression',color=colors, width=0.4, align='edge', hatch='*')
        
        traditional_patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='/', label='Traditional')
        generative_patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='*', label='Generative')
        
        plt.ylim(0)
        plt.axhline(y = 1.0, color = 'black', linestyle = '-')
        plt.xticks(rotation=45)
        plt.legend(handles = [traditional_patch,generative_patch])
        plt.savefig(f"./plots/{ME}Ifprs.png", pad_inches=1, bbox_inches='tight')
        plt.clf()



def get_overal_FPR():
    datas = [pd.read_csv('Data/Combined/genAI_combined.csv'), pd.read_csv('Data/Combined/traditional_combined.csv')]

    MEs = {'Anthropic': [], 'OctoAI': [], 'OpenAI': []}
    true_labels = []
    
    for data in datas:
        for ME in MEs:
            MEs[ME].extend(data[f'{ME}_ME_bool'])
        true_labels.extend(data['true_label'])    

    for ME in MEs:
        hold_true =[]
        hold_pred = []
        for true, pred in zip(true_labels, MEs[ME]):
            if pred == pred:
                hold_true.append(true)
                hold_pred.append(pred)
        print(len(hold_pred), ME)
        MEs[ME] = {"FPR": score_calculator({'true': hold_true, 'pred': hold_pred}, score="FPR")}

    print(MEs)
    
def main():
    # make_bounds()
    # add_normalization()
    # comparison_chart_openAI()
    # comparison_chart_other("PerspectiveAI")
    # comparison_chart_other("Google")
    
    # dataset_table(append=False)
    # print()
    # print(pd.read_csv('./Data/OpenAI_flagging_bounds.csv').to_latex(index=False), '\n')
    # print(pd.read_csv('Data/iavg_table.csv').fillna('').astype('str').to_latex(index=False), '\n')
    # print(pd.read_csv('Data/ifpr_table.csv').fillna('').astype('str').to_latex(index=False))
    # print(pd.read_csv('Data/imed_table.csv').fillna('').astype('str').to_latex(index=False))
    # print(pd.read_csv('Data/ifpr_table_sub_identities.csv').fillna('').astype('str').to_latex(index=False))
    # print(pd.read_csv('Data/imed_table_sub_identities.csv').fillna('').astype('str').to_latex(index=False))
    # gendata = pd.read_csv('Data/GenAI_sub_ids.csv').astype('str')
    # print(gendata[[col for col in gendata.columns if 'CI' not in col]].to_latex(index=False))
    # trad_data = pd.read_csv('Data/Traditional_sub_ids.csv').astype('str')
    # print(trad_data[[col for col in trad_data.columns if 'CI' not in col]].to_latex(index=False))
    
    # identity_imedians('Data/Combined/genAI_combined.csv', label_cols=['true_label'], dataset_name='GenAI', score_types=['True label'], other_table=True)
    # identity_ifprs('Data/Combined/genAI_combined.csv', label_cols=['true_label'], dataset_name='GenAI', score_types=['True label'], other_table=True)
    # identity_imedians('Data/Combined/traditional_combined.csv', label_cols=['true_label'], dataset_name='Traditional', score_types=['True label'], other_table=True)
    # identity_ifprs('Data/Combined/traditional_combined.csv', label_cols=['true_label'], dataset_name='Traditional', score_types=['True label'], other_table=True)


    # add_metrics(identity_ifprs)
    add_metrics(identity_imedians)
    
    plt.rcParams.update({'font.size': 48})
    make_ME_med_charts()
    make_ME_fpr_charts()
    get_overal_FPR()
    # make_lw_data()
    pass


def make_lw_data():
    files = ['Data/Combined/genAI_combined.csv', 'Data/Combined/traditional_combined.csv']
    columns = [
        ['dataset_name', 'subset_name', 'text', 'true_label', 'Big_identity', 'Sub_Identities',
         'PG score','word_length', 'PG-13 score', 'S', 'H', 'V', 'HR', 'SH', 'S3', 'H2', 'V2', 'Toxic',
       'has_slur', 'OpenAI_data', 'Anthropic_data', 'OctoAI_data', 'Perspective_data', 'Google_data', 
       'perspective_ME_responses', 'OpenAI_ME_responses', 'OpenAI_ME_bool',
       'OpenAI_normalized', 'Anthropic_ME_bool',
       'Google_ME_responses', 'OctoAI_ME_responses', 'OctoAI_ME_bool'], 
        ['text', 'Sub_Identities', 'Big_identity', 'true_label', 'dataset_name', 'Hate', 'has_slur', 'subset_name', 'Offensive',
       'severe_toxicity', 'obscene', 'sexual_explicit', 'identity_attack', 'insult', 'threat', 'word_length', 'OpenAI_ME_responses',
       'perspective_ME_responses','OctoAI_ME_responses', 'OctoAI_ME_bool', 'Google_ME_responses',
       'OpenAI_ME_bool', 'OpenAI_normalized',
       'Anthropic_ME_bool', 'OpenAI_data', 'Anthropic_data', 
        'OctoAI_data', 'Perspective_data', 'Google_data']
    ]
    data_subsets = ['GenAI', 'Traditional']
    final_data = None
    for file, col, subset in zip(files, columns, data_subsets):
        data = pd.read_csv(file)#[col]

        data['data_type'] = [subset] * data.shape[0]
        if final_data is None:
            final_data = data.copy()
        else:
            final_data = pd.concat([final_data, data])
    final_data = final_data[['dataset_name', 'subset_name', 'data_type','text',  'word_length', 'true_label',
       'Big_identity', 'Sub_Identities', 'PG score', 'PG-13 score',
       'S', 'H', 'V', 'HR', 'SH', 'S3', 'H2','V2','has_slur',
       'Hate', 'Offensive', 'severe_toxicity', 'obscene', 'sexual_explicit', 'identity_attack', 'insult', 'threat', 
       'perspective_ME_responses', 'OpenAI_ME_bool',
       'OpenAI_normalized', 'Anthropic_ME_bool', 'Google_ME_responses', 'OctoAI_ME_bool',
       'OpenAI_data', 'Anthropic_data', 'OctoAI_data', 'Perspective_data', 'Google_data'
       ]]
    col_change = {
        'perspective_ME_responses': ['jigsaw_score', lambda x: ast.literal_eval(x)\
                    ["attributeScores"]["TOXICITY"]["spanScores"][0]["score"]["value"]\
                        if "API error" not in x and "languageNotSupportedByAttributeError" not in x else None],
        'OpenAI_normalized': ['openAI_score', lambda x: max(ast.literal_eval(x).values()) if 'API error' not in x else None],
        'Google_ME_responses' : ['google_score', lambda x: max(ast.literal_eval(x).values()) if 'API error' not in x else None],
    }
    
    for col in col_change:
        print(col)
        final_data[col] = final_data[col].apply(col_change[col][1])
        final_data.rename(columns={col: col_change[col][0]}, inplace=True)

    final_data.rename(columns={'OpenAI_ME_bool': 'openAI_flag', 'OctoAI_ME_bool': 'llama_guard_flag', 'Anthropic_ME_bool': 'anthropic_flag', 'Perspective_data': 'Jigsaw_data'}, inplace=True)
    
    final_data.to_csv('Data/Combined/Data.csv', index=True)
    
    
if __name__ == "__main__":
    main()
    pass