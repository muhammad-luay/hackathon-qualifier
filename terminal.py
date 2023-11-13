import pandas as pd
from surprise import Dataset, Reader, SVD

# Define a function to clean and convert ratings to a float type
def clean_rating(rating):
    # Dictionary to translate textual numbers to their numeric equivalents
    text_to_num = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}
    
    # Convert rating to string in case it's not
    rating = str(rating)
    
    # Check for textual numbers and convert
    rating = rating.lower()
    if rating in text_to_num:
        return text_to_num[rating]
    
    # Remove any non-digit and non-dot characters
    cleaned_rating = ''.join(c for c in rating if c.isdigit() or c == '.')
    
    # Return the cleaned rating if it's not empty
    return float(cleaned_rating) if cleaned_rating else None


# Function to update or add a user's ratings to the DataFrame
def update_user_ratings(user_ratings, df, user_id=None):
    if user_id is None:
        # If no user_id is provided, assume it's a new user and create a new unique ID
        user_ids = [int(uid) for uid in df['User'].unique() if uid.isdigit()]
        user_id = max(user_ids) + 1 if user_ids else 1
    else:
        # If a user_id is provided, remove their existing ratings from the DataFrame
        df = df[df['User'] != user_id]
    
    # Add the new or updated user ratings to the dataframe
    user_ratings_df = pd.DataFrame({
        'User': [str(user_id)] * len(user_ratings),
        'Movie': list(user_ratings.keys()),
        'Rating': list(user_ratings.values())
    })
    
    # updated_df = df.append(user_ratings_df, ignore_index=True)
    updated_df = pd.concat([df, user_ratings_df], ignore_index=True)

    
    return updated_df, str(user_id)

# Function to recommend a movie
def recommend_movie(user_ratings, df, algo, user_id=None):
    # Update the user's ratings in the DataFrame and get the user ID
    df_updated, user_id = update_user_ratings(user_ratings, df, user_id)
    
    # Load the updated dataset into Surprise
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df_updated[['User', 'Movie', 'Rating']], reader)
    trainset = data.build_full_trainset()
    algo.fit(trainset)
    
    # Get a list of all movies in the dataset
    all_movies = df['Movie'].unique()
    
    # Predict the rating for each movie the user hasn't rated and recommend the highest rated one
    predictions = []
    for movie in all_movies:
        if movie not in user_ratings:
            predicted_rating = algo.predict(user_id, movie).est
            predictions.append((movie, predicted_rating))
            
    # Sort the predictions based on the predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[0][0]

# The main interactive function
def main():
    # Load your dataset here
    df = pd.read_csv('/Users/muhammadluay/Desktop/capital/data.txt')
    
    df['Rating'] = df['Rating'].apply(clean_rating)

    mean_rating = df['Rating'].mean()
    df['Rating'].fillna(mean_rating, inplace=True)

    # Initialize the SVD algorithm
    algo = SVD()

    # List of users
    

    # List of movies
    movies = df['Movie'].unique().tolist()
    movies_str = ', '.join(movies)
    print("List of movies:")
    for i, movie in enumerate(movies, 1):
        print(f"{i}. {movie}")
    print()
    # Ask if the user is new or returning
    is_new_user = input("Are you a new user? (yes/no): ").strip().lower()
    print()
    user_id = None
    if is_new_user == 'no':
        users = df['User'].unique().tolist()
        print("List of users:")
        for i, user in enumerate(users, 1):
            print(f"{i}. {user}")
        
        user_id = input("Please enter your user ID: ").strip()

    # Get movie ratings from the user
    user_ratings = {}
    while True:
        movie_number = input(f"Enter the number of the movie you've seen (or type 'done' to finish): ").strip()
        if movie_number.lower() == 'done':
            break
        if movie_number.isdigit() and 1 <= int(movie_number) <= len(movies):
            movie = movies[int(movie_number) - 1]
            rating = input(f"How would you rate '{movie}' on a scale of 1 to 5?: ").strip()
            if rating.isdigit() and 1 <= int(rating) <= 5:
                user_ratings[movie] = int(rating)
            else:
                print("Invalid rating. Please enter a number from 1 to 5.")
        else:
            print(f"Invalid movie number. Please enter a number from 1 to {len(movies)}.")

    # Recommend a movie based on the user's ratings
    if user_ratings:
        recommended_movie = recommend_movie(user_ratings, df, algo, user_id=user_id)
        print()
        print(f"We recommend you watch: {recommended_movie}")
    else:
        print("No ratings provided. Cannot generate a recommendation.")

if __name__ == "__main__":
    main()
