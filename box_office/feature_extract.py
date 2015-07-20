# -*- coding: utf-8 -*-
"""
Extract data features and labels. Filtering and cleaning is performed on 
the raw data.

Created on Wed Jun 17 16:06:44 2015
@author: ryantonini
"""
from __future__ import division
from collections import OrderedDict
import bisect 

import pandas as pd
import numpy as np 
import imdb 

import data_extract as de


# movie data for following years
YEARS = [2000, 2001, 2002, 2003, 2004, 2005, 2006] 

# mpaa categories
MPAA = ["G", "PG", "PG-13", "R", "NC-17"]

# genres
GENRES = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Fantasy", 
          "Horror", "Romance", "Sci-Fi", "Thriller"]

# initial extension for sequel films page        
INITIAL_EXT ="/wiki/Category:Sequel_films"

# feature names
FEATURES = ["sequel", "budget", "mpaa_G", "mpaa_PG", "mpaa_PG-13", "mpaa_R", "mpaa_NC-17", 
            "director_low", "director_medium", "director_high", "actor_low", 
            "actor_medium", "actor_high", "action", "adventure", "animation", 
            "comedy", "crime", "drama", "fantasy", "horror", "romance", "sci-fi",
            "thriller"]

# revenue edge points in SN
REV_CATEGORIES = [2.5e7, 3.5e7, 4.5e7, 6.5e7, 1e8, 1.5e8, 2e8]
                  
# thresholds for determining director value 
HIGH_DV = 7e7
LOW_DV = 3e7

# thresholds for determining star value 
HIGH_AC = 6e7
LOW_AC = 2.5e7

DIGITS = "0123456789"

MOVIES_LIST = de.get_trainingmovies(YEARS)
SEQUELS_LIST = de.get_sequels(INITIAL_EXT)

def create_moviedf(movies_list, write_to=False):
    """Create a movie dataframe containing each movie in movies list with 
    its associated features.  The dataframe also includes a column containing 
    the target values for each movie.  We write the data to a csv file if 
    write_to is True. 
    
    ***Note if movies_list is large, unexpected errors may arise.  The issue has 
    yet to be resolved but likely resides within the IMDbPY package.  
    Recommeneded usage len(movies_list) <= 40.***
    
    :param movies_list: collection of movie names
    :type movies_list: list of str
    :param write_to: if True write data to csv file
    :type write_to: boolean
    :returns: table of movies with feature, target data
    :rtype: dataframe 
    """
    new_col = FEATURES + ["box office"] # add target column
    df = pd.DataFrame(columns = new_col)
    imdb_access = imdb.IMDb()
    for movie in movies_list:
        if not "(" in movie:
            new = imdb_access.search_movie(movie)[0]
            imdb_access.update(new)
            imdb_access.update(new, "business")
            if (new.has_key("director") and new.has_key("cast") and new.has_key("genre") and 
                    new.has_key("mpaa") and new["business"].has_key("budget") and 
                    new["business"].has_key("gross")):        
               features = []
               features.append(is_sequel(new)) # add sequel info 
               features.append(get_budget(new)) # add budget info
               features += get_mpaa(new) # add mpaa info
               features += get_directorvalue(new, imdb_access) # add director value info
               features += get_starvalue(new, imdb_access) # add actor value info
               features += get_genres(new) # add genre info
               features.append(discretize_revenue(get_revenue(new))) # add box office revenue
               df.loc[new['title']] = features
    budget = np.array(df['budget'])
    df['budget'] = normalize_feature(budget) # update budget values with normalizaed values
    if write_to:
        df.to_csv("movie_dataset.csv", index_label=False)
    return df          

def normalize_feature(column):
    """Gaussian normalization to the elements of column"""
    return (column - np.mean(column))/np.std(column) 
    
def discretize_revenue(revenue):
    """Get the class label corresponding to the revenue given.  
    
    The class intervals for box office revenue are shown below:
        
        1: revenue <= 25 million
        2: 25 million < revenue <= 35 million 
        3: 35 million < revenue <= 45 million
        4: 45 million < revenue <= 65 million
        5: 65 million < revenue <= 100 million
        6: 100 million < revenue <= 150 million 
        7: 150 million < revenue <= 200 million 
        8: revenue > 200 million 
    
    :param revenue: gross box office of a movie in USD
    :type revenue: int
    :returns: class label
    :rtype: int
    """
    return bisect.bisect_left(REV_CATEGORIES, revenue) + 1
        
def get_revenue(movie):
    """Get revenue information (US gross box office) of the movie.
    
    :param movie: a movie from imdb
    :type movie: movie 
    :returns: the gross box office in USD
    :rtype: float
    """
    business_data = movie['business']
    revenue = business_data['gross'][0]
    if "$" in revenue and "(" in revenue:
        ind1 = revenue.index("$")
        ind2 = revenue.index("(")
        revenue = revenue[ind1+1:ind2-1] # remove date, $ sign, and other non-relevant info
    return  int(revenue.replace(",", "")) # convert from str to int 
    
def get_budget(movie):
    """Get budget information (in USD) of the movie.
    
    :param movie: a movie from imdb
    :type movie: movie 
    :returns: the movie budget in USD
    :rtype: float
    """
    business_data = movie['business']
    budget = business_data['budget'][0]
    if '$' in budget:
        ind1 = budget.index("$") + 1
    else:
        for ch in budget: 
            if ch in DIGITS: # find the first digit occuring in budget
                ind1 = budget.index(ch)   
                break     
    if "estimated" in budget:
        ind2 = budget.index("(") 
        budget = budget[ind1:ind2-1] # remove 'estimated', '$' sign and other irrelvant info
    else:
        budget = budget[ind1:len(budget)] # remove '$' sign and other irrelvant info
    return int(budget.replace(",", "")) # remove commas, convert from str to int
       
def get_mpaa(movie):
    """Determine the movie mpaa rating.

    :param movie: a movie from imdb
    :type movie: movie 
    :returns: list of binary coded values for each mpaa category
    :rtype: list of float
    """
    ratings = OrderedDict((rate, 0) for rate in MPAA) # perserves order
    mpaa = movie['mpaa'].split(" ")[1]
    if mpaa in ratings.keys():
        ratings[mpaa] = 1  # quick indexing rather than multiple if/elif's
    return ratings.values() 
    
def get_directorvalue(movie, imdb_access):
    """Determine the director star value in movie.  The three classes of director
    value are: High, Medium, and Low.  The criteria is based on all directors 
    of the movie.

    :param movie: a movie from imdb
    :type movie: movie 
    :param imdb_access: enables access to IMDb's data through the web
    :type imdb_access: IMDbHTTPAccessSystem 
    :returns: list of binary coded values specificing director star value in movie.
    :rtype: list of int  
    """
    total_revenue = 0
    num_movies = 0
    directors = movie['director']
    for person in directors:
        imdb_access.update(person)
        movies = person['director movie'] # get all movies by the director
        for m in movies:
            if 'year' in m.keys() and m['year'] <= movie['year'] and m != movie: # verify movie occured before arg one
                imdb_access.update(m, 'business')
                if m['business'].has_key('gross'): # ensure gross is key
                    value = m['business']['gross'][0]
                    if "USA" in value and "$" in value: # ensure gross is USD, and has $ sign 
                        total_revenue += get_revenue(m)
                        num_movies += 1
    # to avoid division by zero if num_movies = 0
    if num_movies:
        average = total_revenue/num_movies
    else:
        average = 0
    return [int(average < LOW_DV), int(LOW_DV <= average <= HIGH_DV),
            int(average > HIGH_DV)]
                        
def get_starvalue(movie, imdb_access):
    """Determine the actor/actresses star value in movie.  The three classes of 
    star value are: High, Medium, and Low.  The criteria is based on the main
    cast member in the movie. 

    :param movie: a movie from imdb 
    :type movie: movie 
    :param imdb_access: enables access to IMDb's data through the web
    :type imdb_access: IMDbHTTPAccessSystem object
    :returns: list of binary coded values specificing star value in movie.
    :rtype: list of int  
    """
    total_revenue = 0
    num_movies = 0
    person = movie['cast'][0]
    imdb_access.update(person)
    if "actor" in person.keys():
        movies = person['actor'] # get all movies by the actor
    else:
        movies = person['actress'] # get all movies by the actress
    for m in movies:
        if 'year' in m.keys() and m['year'] <= movie['year'] and m != movie: # verify movie occured before arg one
            imdb_access.update(m, 'business')
            if m['business'].has_key('gross'): # ensure gross is a key
                value = m['business']['gross'][0]
                if "USA" in value and "$" in value: # ensure gross is USD, and has $ sign 
                    total_revenue += get_revenue(m)
                    num_movies += 1
    # to avoid division by zero if num_movies = 0
    if num_movies:
        average = total_revenue/num_movies
    else:
        average = 0
    return [int(average < LOW_AC), int(LOW_AC <= average <= HIGH_AC),
            int(average > HIGH_AC)]
     
def get_genres(movie):
    """Determine the movie genres.

    :param movie: a movie from imdb
    :type movie: movie 
    :returns: list of binary coded values for each genre category
    :rtype: list of int
    """
    mg = OrderedDict((genre, 0) for genre in GENRES) # perserves order
    genres = movie['genre']
    for genre in genres:
        if genre in mg.keys():
            mg[genre] = 1  # quick indexing rather than multiple if/elif's
        else:
            continue
    return mg.values()
               
def is_sequel(movie):
    """Determine if the movie is a sequel.

    :param movie: a movie from imdb
    :type movie: movie 
    :returns: true (1) or false (0)
    :rtype: int 
    """
    if movie['title'] in SEQUELS_LIST:
        return 1
    else:
        return 0

