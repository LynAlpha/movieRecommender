import numpy as np
import pandas as pd
import json
import ast

meta = pd.read_csv('movie_dataset/movies_metadata.csv')

meta.head()

meta = meta[['id','original_title','original_language','genres']]
meta = meta.rename(columns={'id':'movieId'})
meta = meta[meta['original_language'] == 'en']
meta.head()

ratings = pd.read_csv('movie_dataset/ratings_small.csv')
ratings = ratings[['userId','movieId','rating']]
ratings.head()

ratings.describe()

meta.movieId = pd.to_numeric(meta.movieId,errors='coerce')
ratings.movieId = pd.to_numeric(ratings.movieId,errors='coerce')

def parse_genres(genres_str):
    genres = json.loads(genres_str.replace('\'','"'))

    genres_list = []
    for g in genres:
        genres_list.append(g['name'])

    return genres_list

meta['genres'] = meta['genres'].apply(parse_genres)

data = pd.merge(ratings,meta,on='movieId',how='inner')

data.head()

matrix = data.pivot_table(index='userId',columns='original_title',values='rating')
matrix_u = data.pivot_table(index='original_title',columns='userId',values='rating')
matrix.head(20)

GENRE_WEIGHT = 0.1

def pearsonR(s1, s2):
    s1_c = s1 - s1.mean()
    s2_c = s2 - s2.mean()
    return np.sum(s1_c * s2_c) / np.sqrt(np.sum(s1_c ** 2) * np.sum(s2_c ** 2))

def movie_based(input_movie, matrix, n, similar_genre=True):
    input_genres = meta[meta['original_title'] == input_movie]['genres'].iloc(0)[0]

    result = []
    for title in matrix.columns:
        if title == input_movie:
            continue

        # rating comparison
        cor = pearsonR(matrix[input_movie], matrix[title])
        
        # genre comparison
        if similar_genre and len(input_genres) > 0:
            temp_genres = meta[meta['original_title'] == title]['genres'].iloc(0)[0]

            same_count = np.sum(np.isin(input_genres, temp_genres))
            cor += (GENRE_WEIGHT * same_count)
        
        if np.isnan(cor):
            continue
        else:
            result.append(title)
            
    return result[:n]

def user_based(input_user, matrix_u, n):
    result = []
    for user in matrix_u.columns:
        if user == input_user:
            continue

        cor = pearsonR(matrix_u[input_user], matrix_u[user])

        if np.isnan(cor):
            continue
        else:
            result.append(user)

    return result[:n]


def user_rating_based(input_user):
    likedmovie = []
    for titidx in range(len(matrix.loc[input_user])):
        if np.isnan(matrix.loc[input_user][titidx]) is True:
            continue
        elif matrix.loc[input_user][titidx] >= 3.0:
            likedmovie.append(str(matrix.columns[titidx]))

    movielist = []
    for tit in likedmovie:
        movielist.extend(movie_based(tit, matrix, 10, similar_genre=True))
    movielist = [x for x in movielist if x not in likedmovie]
    return movielist

   
def recom(input_user):
    movies = user_rating_based(input_user)
    simusers = user_based(input_user, matrix_u, 1)
    for friend in simusers:
        for titidx in range(len(matrix.loc[friend])):
            if np.isnan(matrix.loc[friend][titidx]) is True:
                continue
            elif matrix.loc[friend][titidx] >= 3.0:
                movies.append(str(matrix.columns[titidx]))
    temp =[]
    
    for t in movies:
        if t not in temp:
            temp.append(t)
    print('Recommend movies')
    print(temp)
    return 
