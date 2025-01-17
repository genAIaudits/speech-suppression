""""
Description: A program to use the wikipedia API to gather TV and movie data
Author: Charlie Crawford + Fejiro Anigboro
Date:
Created for use in the GPT benchmark suite, more information here: 
"""

# Imports:
import wikipedia as wiki
import pandas as pd
import tmdbsimple as tmdb 
import optparse
import sys, os
import numpy as np
import time
# https://www.geeksforgeeks.org/python-import-from-parent-directory/
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
# adding the parent directory to 
# the sys.path.
sys.path.append(parent)
from main import conv_openAI_ME_data
from run_openai_ME import run_me_caller
from run_perspective_ME import run_Perspective_ME
from run_llama_ME import run_OctoAI_ME
from run_google_ME import run_google_ME
from run_anthropic_ME import run_anthropic_ME
from movie_tagger import run_audit as more_data_prep
"""
API Key
"""
tmdb.API_KEY = 'ENTER API KEY'


def data_prep():
    opts = parse_args()
    out_data = pd.DataFrame()
    try:
        dataset = pd.read_csv(opts.dataset_filename)
        colname = opts.col_name
    except: # TODO - make this more specific 
       print("Dataset not found")
       return

    # tmbd with release year and genre
    genre, year, overview = get_genre_year(dataset["id"])
    dataset["genres"] = genre
    dataset["Release year"] = year
    dataset.to_csv("./Data/Movies/TMDB_with_genre.csv", index=False)
    
    # adding overviews
    dataset["Overview"] = overview
    dataset.to_csv("./Data/Movies/TMDB_with_overview.csv", index=False)
    
    print("Getting show IDs...")
    ids = get_tmdb_ids(dataset, True, colname) # returns a set
    # turn list of ids into Series, append to df 

    print("Getting content ratings...")
    names, ratings, release_dates = get_content_ratings(ids, True)
    # turn list of ratings into Series, append to df 
    out_data['title'] = names
    out_data['ratings'] = ratings
    out_data['release'] = release_dates


    # getting wiki plots
    print("Getting wiki descriptions...")
    descs = get_wiki_descs(dataset) # change back to out_data 
    dataset['plots'] = descs
    dataset.to_csv('./Data/Movies/full_TMDB_with_wiki_final.csv', index=False)

    print("Getting rating scores...")
    print(dataset.shape)
    dataset.dropna(subset=["plots"], inplace=True)
    print(dataset.shape)
    ratings_set = ['G', 'PG', 'PG-13', 'R', 'NC-17']
    toxicity_scores = score_toxicity(dataset, ratings_set, 'rating')
    print("Appending rating scores...")
    for i in range(len(ratings_set)):
        dataset[f"{ratings_set[i]} score"] = toxicity_scores[:, i]
    print("Finished appending scores")
    dataset.to_csv('./Data/Movies/full_TMDB_wiki_with_scores.csv', index=False)


def get_ME_responses():
    more_data_prep()
    dataset = pd.read_csv("./Data/Movies/TMDB_plots_scores_identity.csv")

    # moderation endpoint calls
    print("Calling OpenAI ME on movie data")
    print("Getting ME responses...")
    dataset = dataset.dropna(subset=["plots"]).reset_index()
    responses = run_me_caller(dataset, 'plots')
    print("Finished calling ME ")
    dataset['OpenAI_ME_responses'] = responses[0]
    print("Converting ME_responses")
    dataset["OpenAI_ME_bool"] = conv_openAI_ME_data(responses[0])
    dataset['OpenAI_data'] = responses[1]
    dataset.to_csv('./Data/Movies/TMDB_with_ME.csv', index=False)
    
    # perspective AI call
    start = time.time()
    perspective_responses = run_Perspective_ME(dataset['plots'].tolist())
    dataset["perspective_ME_responses"] = perspective_responses[0]
    dataset['Perspective_data'] = perspective_responses[1]
    print("Elapsed time:", time.time() - start)
    dataset.to_csv('./Data/Movies/TMDB_with_ME.csv', index=False)

    # OctoAI call
    start = time.time()
    OctoAI_responses = run_OctoAI_ME(dataset['plots'].tolist())
    dataset["OctoAI_ME_responses"] = OctoAI_responses[0]
    dataset["OctoAI_ME_bool"] = OctoAI_responses[1]
    dataset['OctoAI_data'] = OctoAI_responses[2]
    print("Elapsed time:", time.time() - start)
    dataset.to_csv('./Data/Movies/TMDB_with_ME.csv', index=False)

    # Google ME call
    start = time.time()
    Google_responses = run_google_ME(dataset['plots'].tolist())
    dataset["Google_ME_responses"] = Google_responses[0]
    dataset['Google_data'] = Google_responses[1]
    print("Elapsed time:", time.time() - start)
    dataset.to_csv('./Data/Movies/TMDB_with_ME.csv', index=False)
    
    # Anthropic ME call
    start = time.time()
    Anthropic_responses = run_anthropic_ME(dataset['plots'].tolist())
    dataset["Anthropic_ME_responses"] = Anthropic_responses[0]
    dataset["Anthropic_ME_bool"] = Anthropic_responses[1]
    dataset['Anthropic_data'] = Anthropic_responses[2]
    print("Elapsed time:", time.time() - start)
    dataset.to_csv('./Data/Movies/TMDB_with_ME.csv', index=False)


def run_movies_audit():
    data_prep()
    get_ME_responses()

    
def parse_args():
    """Parse command line arguments."""
    parser = optparse.OptionParser(description='run moderation endpoint script')

    parser.add_option('-d', '--dataset_filename', type='string', help='path to' +\
        ' dataset file (CSV format)')
    parser.add_option('-c', '--col_name', type='string', help='name of' +\
        ' column in dataset to run ME on')

    (opts, args) = parser.parse_args()

    mandatories = ['dataset_filename',]
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    return opts


def get_tmdb_ids(df, is_movie, colname):
    ids = []
    search = tmdb.Search() # may have to move this 

    # movie case
    if is_movie:
        for title in df[colname]:
            response = search.movie(query=title)
            max_popular = 0
            for s in search.results:
                if s['title'] == title and s['original_language'] == 'en':
                    if int(s['popularity']) > max_popular:
                        # movie_id = s['id']
                        # print(s['title'], movie_id)
                        # ids.append(movie_id)
                        max_popular = int(s['popularity'])
                        movie = s
                # else:
                #     # print("Title not found in TMDB")
                #     ids.append("")
                #     break
            try:
                movie_id = movie['id']
                ids.append(movie_id)
            except UnboundLocalError as ule:
                print("Title " + str(title) +" not found in TMDB")
                ids.append("")
            
    # tv case - need to handle this differently depending on how ids are made (by season? by show?)
    else:
        for title in df['title']:
            pass
    
    return set(ids)


def read_tmdb_csv(filepath):
    tmdb_data = pd.read_csv(filepath)
    return tmdb_data


def get_genre_year(ids):
    release_dates = []
    genres = []
    overviews = []
    
    for id in ids:
        try:
            if id != '':
                movie = tmdb.Movies(id)
                response = movie.info()
                curr_movie_genres = response["genres"]
                all_genres = [genre["name"] for genre in curr_movie_genres]
                if int(response['release_date'][:4]) > 1985:
                    genres.append(";".join(all_genres))
                    release_dates.append(response['release_date'][:4])
                    overviews.append(response["overview"])
            else:
                genres.append("")
                overviews.append("")
                release_dates.append("")
        except:
            print(id)
            pass
    assert(len(release_dates) == len(genres) == len(overviews))
    return genres, release_dates, overviews  


def get_content_ratings(ids, is_movie):
    ratings = []
    names = []
    release_dates = []

    if is_movie:
        for id in ids:
            if id != '':
                cr = ""
                name = ""
                movie = tmdb.Movies(id)
                response = movie.info()
                release = movie.releases()['countries']
                for i in range(len(release)):
                    if release[i]['iso_3166_1'] == 'US' and release[i]['certification'] != '':
                        cr = release[i]['certification']
                        name = movie.title
                        print(cr, name)
                        break
                    
                if cr == "":
                    print("No US ratings available for " + str(movie.title))
                ratings.append(cr)
                names.append(name)
                release_date = response['release_date'][:4]
                release_dates.append(release_date)
    
    return names, ratings, release_dates


def help_search(page_title:str, movie_title:str, layer:int) -> tuple:
    """
    Helper function to get plot from wiki page
    Args:
        page_title (str): title of wiki page 
        movie_title (str): title of movie
        layer (int): layer of plot search

    Returns:
        tuple[str, bool]: the plot and bool to represent if search is complete
    """
    finish_search = False
    page = wiki.page(page_title, auto_suggest=False)
    plot = page.section('Plot')

    if plot != "" and plot is not None:
        finish_search = True
        print(f"{movie_title} synopsis found via Wikipedia layer {layer} search!")
    
    return plot, finish_search


def get_wiki_descs(df: pd.DataFrame) -> list:
    """
    A function to query the Wikipedia API for plot descriptions of movies provided 
    Input: a dataframe with at least cols: 'title' and 'release' 
    Output: a list containing plots for movies in input df
    """
    plots = []
    found = 0
    not_found = 0
    titles = list(df["Title"])
    years = list(df["Release year"])
    
    for title, release_year in zip(titles, years):
        finish_search = False
        if title != "":
            plot = ""
            try:
                plot, finish_search = help_search(page_title=title, movie_title=title, layer=1)
            except Exception as e:
                print(f"Error in Wikipedia layer 1 for {title}: {type(e)}")
            
            if not finish_search:
                try: # try searching with title + "film" 
                    plot, finish_search = help_search(page_title=title + " (film)", 
                                            movie_title=title, layer=2)
                except Exception as e:
                    print(f"Error in Wikipedia layer 2 for {title}: {type(e)}")
                
            if not finish_search:
                try: # try searching with year + "film" 
                    plot, finish_search = help_search(page_title=title + " (" + str(release_year) + " film)", 
                                            movie_title=title, layer=3)
                except Exception as e:
                    print(f"Wikipedia layer 3 error: {type(e)}, ending search.")
                    
            if not finish_search:
                try: # try searching with year + "American film" 
                    plot, finish_search = help_search(page_title=title + " (" + str(release_year) + " American film)", 
                                            movie_title=title, layer=4)
                except Exception as e:
                    print(f"Wikipedia layer 4 error: {type(e)}, ending search.")
                
            if not finish_search:
                try: # try searching with title + "American film" 
                    plot, finish_search = help_search(page_title=title + " (American film)", 
                                            movie_title=title, layer=5)
                except Exception as e:
                    print(f"Error in Wikipedia layer 5 for {title}: {str(e)}")
                  
                    
            if plot == "" or plot is None:
                not_found += 1
            else:
                found += 1
            plots.append(plot)
            
    print("Not found: "+ str(not_found))
    print("Found "+ str(found))
    
    return plots 


def score_toxicity(df:pd.DataFrame, ratings_ordered_set:list, col:str) -> np.ndarray:
    """
    Scores the toxicity of each movie based on its rating.
    Ratings ordered in ascending order. ['G', 'PG', 'PG-13', 'R', 'NC-17']
    Movie with a rating or below has a 0 for toxic, and ratings above have 1.
    i.e. movie rated PG-13 has [1 1 1 0 0] 
    i.e. movie rated R has [1 1 1 1 0]

    Args:
        df (pandas dataframe): dataframe containing movie title and other relevant info.
        only care about the 'rating' column here
        ratings_ordered_set (list): ratings list
        col (str): col name of the 

    Returns:
        np.ndarray: np array with each row containing scores (yes/no) for each movie per rating
    """
    toxicity_scores = []
    movie_ratings = list(df[col])

    for rating in movie_ratings:
        current_ratings = []
        rating_loc = ratings_ordered_set.index(rating)
        
        for i in range(len(ratings_ordered_set)):
            current_ratings.append(0) if i >= rating_loc else current_ratings.append(1)
            
        toxicity_scores.append(current_ratings)
   
    return np.array(toxicity_scores) 


if __name__ == "__main__":
    run_movies_audit()