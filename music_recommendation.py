# Install the Surprise library if not already installed
# pip install scikit-surprise

import pandas as pd
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import train_test_split, cross_validate
from surprise import accuracy

# Step 1: Load the Uploaded Dataset
file_path = "music_dataset.csv"
uploaded_data = pd.read_csv(file_path)

print("Sample Dataset:")
print(uploaded_data.head())

# Step 2: Prepare the Dataset for Surprise
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(uploaded_data[['user_id', 'song_id', 'play_count']], reader)

# Step 3: Train-Test Split
trainset, testset = train_test_split(dataset, test_size=0.2)

# Step 4: Train the Model
model = SVD()
model.fit(trainset)

# Step 5: Evaluate the Model
predictions = model.test(testset)
print("\nModel Accuracy:")
accuracy.rmse(predictions)

# Step 6: Cross-Validation
print("\nCross-Validation Results:")
cross_validate(model, dataset, measures=['RMSE', 'MAE'], cv=3, verbose=True)

# Step 7: Recommend Songs for a User
def recommend_songs(user_id, data, model, top_n=5):
    """
    Recommend top N songs for a given user_id.
    """
    all_songs = data['song_id'].unique()
    user_songs = data[data['user_id'] == user_id]['song_id'].unique()
    not_listened = [song for song in all_songs if song not in user_songs]

    # Predict ratings for songs the user has not listened to
    predictions = [model.predict(user_id, song) for song in not_listened]
    predictions.sort(key=lambda x: x.est, reverse=True)

    top_recommendations = predictions[:top_n]
    return [(rec.iid, rec.est) for rec in top_recommendations]

# Example Usage: Recommend Songs for a Random User
random_user = uploaded_data['user_id'].sample().iloc[0]
print(f"\nRecommendations for {random_user}:")
recommendations = recommend_songs(random_user, uploaded_data, model)
for song, score in recommendations:
    print(f"Song: {song}, Predicted Score: {score}")