import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Function to read CSV file from fixed destination
def read_csv_from_destination(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error: Failed to read CSV file '{file_path}': {e}")
        return None

# Function to preprocess data and train Random Forest classifier
def train_random_forest_classifier(df, target_column):
    # Drop duplicates, dropna, and remove 'Timestamp' column
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df.drop(columns=['Timestamp'], inplace=True)

    # Encode categorical columns
    label_encoder = LabelEncoder()
    one_hot_encoder = OneHotEncoder(sparse=False)

    # Encode non-numeric columns
    for column in df.select_dtypes(include=['object']).columns:
        if column != target_column:
            df[column] = label_encoder.fit_transform(df[column])

    # Define input features (X) and target variable (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split the dataset into training and testing sets
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    return rf_classifier

# Streamlit UI
def main():
    st.title("Random Forest Classifier")

    # Read CSV file from fixed destination
    file_path = "C:/Shreyas/intern/data/ad_final.csv"
    target_column = "Clicked on Ad"
    df = read_csv_from_destination(file_path)

    if df is not None:
        # Train Random Forest classifier
        rf_classifier = train_random_forest_classifier(df, target_column)

        # User input for prediction
        st.subheader("Enter values for prediction:")
        user_inputs = {}
        for column in df.columns:
            if column == target_column:
                # Make 'Clicked on Ad' read-only
                user_input = df[target_column].iloc[0]
            elif column == 'Ad Topic Line':
                # Use select box for 'Ad Topic Line' with future name and one-hot encoded value
                user_input = st.selectbox(f"Select value for {column}", df[column].unique())
            elif column == 'Country':
                # Use select box for 'Ad Topic Line' with future name and one-hot encoded value
                user_input = st.selectbox(f"Select value for {column}", df[column].unique())
            elif column == 'City':
                # Use select box for 'Ad Topic Line' with future name and one-hot encoded value
                user_input = st.selectbox(f"Select value for {column}", df[column].unique())
            elif column == 'Gender':
                # Use select box for 'Ad Topic Line' with future name and one-hot encoded value
                user_input = st.selectbox(f"Select value for {column}", df[column].unique())
            else:
                # Use select box for other numeric columns
                user_input = st.text_input(f"Enter value for {column}", "")
            user_inputs[column] = user_input

        if st.button("Get Prediction"):
            # Prepare user input as DataFrame for prediction
            user_df = pd.DataFrame(user_inputs, index=[0])

            # Reindex user input DataFrame to match feature names used during training
            user_df = user_df.reindex(columns=df.drop(columns=[target_column]).columns, fill_value=0)

            # Make prediction
            prediction_proba = rf_classifier.predict_proba(user_df)
            predicted_class = rf_classifier.predict(user_df)

            st.write("Predicted Probabilities:", prediction_proba)
            st.write("Predicted Class:", predicted_class)

            # Determine the action based on predicted probability
            if prediction_proba[0][0] > 0.75:
                st.write("Repeat the ad two more times.")
            elif 0.5 <= prediction_proba[0][0] <= 0.75:
                st.write("Repeat the ad once.")
            else:
                st.write("Change the ad to something similar to the current ad.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
