# IMPORTING LIBRARIES

from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# READING DATASETS
Games = pd.read_csv("Clean_Data/Games.csv")
Reviews = pd.read_csv("Clean_Data/Reviews.csv")
Items = pd.read_parquet("Clean_Data/Items.parquet")


# INITIALIZING APP
app = FastAPI()


# CREATING ROUTES FOR OUR API

@app.get('/PlayTimeGenre/{genre}')
async def PlayTimeGenre(genre: str):
    
    try:
        value = genre
        results = Items[Items['tags_and_genres'].str.contains(value, case=False, na=False)]

        if results.empty:
           return "Invalid genre."
        
        results_2 = results.groupby('item_id')['playtime_forever'].sum().reset_index()
        results_2
        
        merged_df = pd.merge(results_2, Games[['id', 'release_date']], left_on='item_id', right_on='id', how='left')
        
        playtime = merged_df[merged_df['playtime_forever'] == merged_df['playtime_forever'].max()]

        release_date = playtime['release_date'].values[0]

        return int(release_date)

    except Exception as e:
        return {"message": f"Error: {str(e)}"}

@app.get('/UserForGenre/{genre}')
async def UserForGenre(genre: str):

    try:
        value = genre
        results = Items[Items['tags_and_genres'].str.contains(value, case=False, na=False)]
        if results.empty:
           return "Invalid genre."

        results_2 = results.groupby('user_id')['playtime_forever'].sum().reset_index()
        results_2 = results_2.sort_values(['playtime_forever'], ascending= False)
        looking_for_user = results_2.iloc[0]

        results_3 = Items[(Items['user_id'] == looking_for_user['user_id']) & (Items['tags_and_genres'].str.contains(value, case=False, na=False))]

        results_3 = results_3.sort_values(['playtime_forever'], ascending= False)
      
        results_4 = pd.merge(results_3, Games[['id', 'release_date']], left_on='item_id', right_on='id', how='left')
        results_4 = results_4.groupby('release_date')['playtime_forever'].sum().reset_index()
        results_4 = results_4.sort_values(['release_date'], ascending= False)

        year_hours = [(f"Year: {int(year)}", f"hours: {int(hora)}") for year, hora in zip(results_4['release_date'], results_4['playtime_forever'] / 60) if hora > 0]
        year_hours.insert(0,{"The user with most played hours in genre is: ",looking_for_user['user_id']})

        return(year_hours)

    

    except Exception as e:
        return {"message": f"Error: {str(e)}"}

@app.get('/UsersRecommend/{year}')
async def UsersRecommend(year: int):

    try:
        year_search = int(year)
        recommend_year = Reviews[Reviews['posted'] == year_search]
        if recommend_year.empty:
           return "year invalid."

        recommend_true = recommend_year[recommend_year['recommend'] & ((recommend_year['sentiment_analysis'] == 1) | (recommend_year['sentiment_analysis'] == 2))]

        top3most = recommend_true.groupby('item_id')['sentiment_analysis'].count().reset_index()
        
        top3most = pd.merge(top3most, Games[['id', 'title']], left_on='item_id', right_on='id', how='left')

        top3m = top3most.sort_values(['sentiment_analysis'], ascending= False).head(3)

        list_data_save = [(f'Puesto {i + 1}', top3m['title'].iloc[i]) for i in range(len(top3m))]

        return(list_data_save)

    except Exception as e:
        return {"message": f"Error: {str(e)}"}

@app.get('/UsersNotRecommend/{year}')
async def UsersNotRecommend(year: int):

    try:
        year_search = int(year)
        recommend_year = Reviews[Reviews['posted'] == year_search]
        if recommend_year.empty:
           return "Invalid Year."
    
        recommend_false = recommend_year[(recommend_year['recommend'] == False) & (recommend_year['sentiment_analysis'] == 0)]

        top3least = recommend_false.groupby('item_id')['sentiment_analysis'].count().reset_index()

        top3least = pd.merge(top3least, Games[['id', 'title']], left_on='item_id', right_on='id', how='left')

        top3l = top3least.sort_values(['sentiment_analysis'], ascending= False).head(3)

        list_data_save = [(f'Puesto {i + 1}', top3l['title'].iloc[i]) for i in range(len(top3l))]

        return(list_data_save)

    except Exception as e:
        return {"message": f"Error: {str(e)}"}

@app.get('/sentiment_analysis/{year}')
async def Sentiment_Analysis(year: int):

    try:
        year_search = int(year)

        sentiment_year = Reviews[Reviews['posted'] == year_search]
        sentiment_year = sentiment_year.groupby('sentiment_analysis').size().reset_index(name='count')

        list_data_save = []
        list_data_save.insert(0,"Year realease: "+ str(year_search))
        list_data_save.insert(1, "Negative Result from Sentiment Analysis : " + str(sentiment_year['count'].iloc[0]))
        list_data_save.insert(2, "Neutral Result from Sentiment Analysis : " + str(sentiment_year['count'].iloc[1]))
        list_data_save.insert(3, "Positive Result from Sentiment Analysis : " + str(sentiment_year['count'].iloc[2]))
        return(list_data_save)

    except Exception as e:
        return {"message": f"Error: {str(e)}"}


@app.get('/game_recommendation/{product_id}')
async def game_recommendation(product_id: int):

    try:

        target_game = Games[Games['id'] == int(product_id)]

        num_recommendations = 5

        if target_game.empty:
            return "Game not found."


        target_game_tags_and_genres = ' '.join(target_game['tags'].fillna('') + ' ' + target_game['genres'].fillna(''))


        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(Games['tags'].fillna('') + ' ' + Games['genres'].fillna(''))


        similarity_scores = cosine_similarity(tfidf_vectorizer.transform([target_game_tags_and_genres]), tfidf_matrix)


        similar_games_indices = similarity_scores[0].argsort()[::-1]


        recommended_games = Games.loc[similar_games_indices[1:num_recommendations + 1]]
        recommended_list = recommended_games['title'].tolist()
        

        return recommended_list

    except Exception as e:
        return {"message": f"Error: {str(e)}"}