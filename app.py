import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io
import streamlit as st
from streamlit_option_menu import option_menu

# Set the page layout to wide mode
st.set_page_config(layout="wide")

# Load the dataset
df = pd.read_csv('labeled_data.csv')

# Set Streamlit page title
#st.title('Hate Speech and Offensive Language Analysis')

# Create a vertical tab menu in the sidebar
with st.sidebar:
    selected = option_menu(
        menu_title="Hate Speech and Offensive Language Analysis",  # Title of the menu
        options=["Home", "Tweets Dataset", "Tweets Classes", "Tweets Preprocessing", "Model Selection", "Try The Model", "About", "Contact"],  # Menu options
        icons=["house","","","","","","info","envelope"],  # Optional icons
        menu_icon="cast",  # Icon for the menu title
        default_index=0,  # Default selected option
        orientation="vertical"  # Set the orientation to vertical
    )

# Display content based on selected tab
if selected == "Home":
    st.title("Introduction")
    st.write("""This dataset contains data related to hate speech and offensive language. 
    Davidson introduced a dataset of tweets categorized using a crowdsourced hate speech vocabulary. 
    These tweets were classified into three categories: hate speech, offensive language, and neither. 
    The dataset, consisting of 24,802 labeled tweets, includes columns for the number of CrowdFlower coders, 
    the count of hate speech and offensive language identifications, and a class label indicating 
    the majority opinion: 0 for hate speech, 1 for offensive language, and 2 for neither.\n
    The dataset published in:\n
    Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017, May). Automated hate speech 
    detection and the problem of offensive language. In Proceedings of the international 
    AAAI conference on web and social media (Vol. 11, No. 1, pp. 512-515).
    The Dataset can be downloaded from:
    
    https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset
    
    https://github.com/t-davidson/hate-speech-and-offensive-language
    
    """)
    # Horizontal line separator
    st.markdown("---")

elif selected == "Tweets Dataset":
    st.title("Loading and Previewing the Dataset")
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Dataset Information", "Dataset Description", "Dataset Overview", "Missing values"])

    # Tab 1: Dataset Brief Information
    with tab1:
        st.subheader('Dataset Information')

        # Capture the df.info() output
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()

        # Display the info in Streamlit
        st.text(s)

    # Tab 2: Dataset Columns Description
    with tab2:
        st.subheader('Dataset Columns Description')
        st.write(df.describe(include='all'))

    # Tab 3: Dataset Overview (Before Preprocessing)
    with tab3:
        st.subheader('Dataset Overview (Before Preprocessing)')
        st.write(df.head(10))

    # Tab 4: Check for missing data
    with tab4:
        # Check for missing data
        st.subheader("Missing values in each column:")
        st.write(df.isnull().sum())
   
    # Horizontal line separator
    st.markdown("---")

elif selected == "Tweets Classes":
    st.title("Understanding Class Distribution")
    # Sample Data (replace this with your actual DataFrame)
    # Ensure 'class' is in your DataFrame (0: Hate Speech, 1: Offensive Language, 2: Neither)
    data = {'class': [0, 1, 2, 0, 1, 2, 1, 1, 0, 2, 0, 1, 0, 2, 1]}  # Example data
    df_fig = pd.DataFrame(data)

    # Class labels
    class_labels = ['Hate Speech', 'Offensive Language', 'Neither']

    # Title of the app
    st.title("Understanding Class Distribution")

    # Create tabs
    tab1, tab2 = st.tabs(["Bar Chart", "Pie Chart"])

    # Tab 1: Distribution of Classes (Bar Chart)
    with tab1:
        st.subheader('Distribution of Classes (Bar Chart)')
    
        # Count occurrences of each class
        class_counts = df_fig['class'].value_counts().reindex([0, 1, 2], fill_value=0)

        # Create a bar chart using Plotly
        bar_fig = px.bar(
            x=class_labels, 
            y=class_counts.values, 
            labels={'x': 'Class', 'y': 'Frequency'}, 
            title='Distribution of Classes',
            color=class_labels,
        )
    
        # Show the bar chart
        st.plotly_chart(bar_fig)

    # Tab 2: Proportion of Classes (Pie Chart)
    with tab2:
        st.subheader('Proportion of Classes (Pie Chart)')
    
        # Create a pie chart using Plotly
        pie_fig = go.Figure(
            data=[go.Pie(
                labels=class_labels, 
                values=class_counts.values, 
                hole=0.3,  # Make it a donut chart for style
                pull=[0, 0.1, 0],  # Pull out the second slice slightly
                marker=dict(colors=['#FF6347', '#FFD700', '#90EE90']),
                textinfo='label+percent', 
                hoverinfo='label+value'
            )]
        )
    
        pie_fig.update_layout(
            title_text="Distribution of Classes (Pie Chart)",
            showlegend=True
        )
    
        # Show the pie chart
        st.plotly_chart(pie_fig)
        # Horizontal line separator
        st.markdown("---")

elif selected == "Tweets Preprocessing":
    st.title("Dataset Preprocessing")

    st.write("""
    We used slight pre-processing to normalize the tweets content by:
    A) Delete the characters outlined here (— : , ;	! ?).
    B) Normalize hashtags into words, thus ’refugeesnotwelcome’ becomes ’refugees not welcome’.
       This is due to the fact that such hashtags are frequently employed when creating phrases. 
    C) We separate such hashtags using a dictionary-based lookup.
    D) To eliminate word inflections, use lowercase to remove capital letters and stemming to overcome the problem of several forms of words.
    E) Encode the tweets into integers and pad each tweet to the max length of 100 words.

    """)
    # Horizontal line separator
    st.markdown("---")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Tweets Before Preprocessing", "Cleaned Tweets", "Stemmed Tweets", "Tokenized Tweets"])

    # Tab 1: Tweets Before Preprocessing
    with tab1:
        st.subheader('Tweets Before Preprocessing')
        st.write(df.tweet)
        # Horizontal line separator
        st.markdown("---")

    # Tab 2: Tweets After Cleaning
    with tab2:
        st.subheader('Tweets After Cleaning')
        st.write(pd.read_csv('cleaned_tweets.csv'))
        # Horizontal line separator
        st.markdown("---")

    # Tab 3: Tweets After Stemming
    with tab3:
        st.subheader('Tweets After Stemming')
        st.write(pd.read_csv('stemmed_tweets.csv'))
        # Horizontal line separator
        st.markdown("---")

    # Tab 4: Tweets After Tokenization
    with tab4:
        st.subheader('Tweets After Tokenization')
        st.write(pd.read_csv('Tokenized_Padded_tweets.csv'))
        # Horizontal line separator
        st.markdown("---")

elif selected == "Model Selection":
    st.title("Model Selection")
    st.write("""
    (Classifier training and testing): Ten-fold cross-validation was used to train 
    and test all the six classifiers (logistic regression, decision tree, random forest, 
    naive Bayes, k-nearest neighbors, and support vector machines). We utilized 
    traditional machine learning methods provided by the Scikit-learn Python module 
    for classification. The Logistic Regression class uses L2 regularization with 
    a regularization parameter C equals 0.01. The hyper parameter used value of maximum depth 
    in decision trees and random forest equals 2. The hyper parameter used value of k in 
    k-nearest neighbors is 5, this means that the algorithm will consider the class or value of 
    the 5 nearest neighbors, when making predictions. In naive Bayes there are no specific default 
    values for this algorithm, as it does not require tuning hyper parameters. The hyper parameter 
    used value of C in SVM is 1.0.""")
    # Horizontal line separator
    st.markdown("---")
    tab1, tab2 = st.tabs(["Classification Results", "Display Results Figures"])
    # Tab 3: Table I. Classification Results
    with tab1:
        st.subheader('Table I. Classification Results')
        # Define the data for the table
        data = {
        'Algorithm': ['Logistic Regression', 'Decision Tree', 'Random Forest', 
                      'Naive Bayes', 'K-Nearest Neighbor', 'SVM - SVC'],
        'Precision': ['0.83 ± 0.04', '0.77 ± 0.06', '0.77 ± 0.06', '0.71 ± 0.07', '0.79 ± 0.05', '0.78 ± 0.05'],
        'Recall': ['0.96 ± 0.02', '1.00 ± 0.01', '1.00 ± 0.01', '0.96 ± 0.02', '0.90 ± 0.03', '1.00 ± 0.01'],
        'F1-Score': ['0.88 ± 0.02', '0.87 ± 0.03', '0.87 ± 0.03', '0.81 ± 0.04', '0.84 ± 0.04', '0.87 ± 0.03']
        }

        # Convert the data to a pandas DataFrame
        df_results = pd.DataFrame(data)

        # Display the table in Streamlit
        st.table(df_results)
        # Horizontal line separator
        st.markdown("---")

    # Tab 2: Display Results Figures
    with tab2:
        st.subheader('Display Results Figures')
        # Data for the table
        data = {
            'Algorithm': ['Logistic Regression', 'Decision Tree', 'Random Forest', 
                          'Naive Bayes', 'K-Nearest Neighbor', 'SVM - SVC'],
            'Precision': [0.83, 0.77, 0.77, 0.71, 0.79, 0.78],
            'Recall': [0.96, 1.00, 1.00, 0.96, 0.90, 1.00],
            'F1-Score': [0.88, 0.87, 0.87, 0.81, 0.84, 0.87]
        }

        # Convert the data to a pandas DataFrame (renaming it df_fig)
        df_fig = pd.DataFrame(data)

        # Create a grouped bar chart using Plotly
        fig = go.Figure()

        # Add Precision bars
        fig.add_trace(go.Bar(x=df_fig['Algorithm'], y=df_fig['Precision'], name='Precision'))

        # Add Recall bars
        fig.add_trace(go.Bar(x=df_fig['Algorithm'], y=df_fig['Recall'], name='Recall'))

        # Add F1-Score bars
        fig.add_trace(go.Bar(x=df_fig['Algorithm'], y=df_fig['F1-Score'], name='F1-Score'))

        # Update layout for grouped bars
        fig.update_layout(
            title='Classification Results',
            xaxis_title='Algorithm',
            yaxis_title='Score',
            barmode='group',  # Group the bars side by side
            xaxis_tickangle=-45
        )
        # Display the plot in Streamlit
        st.plotly_chart(fig)
        st.markdown("---")

    st.title("Results Clarifaction")
    st.write("""
    Looking at the results, it appears that the Decision Tree, Random Forest, 
    and SVM - SVC classifiers have the highest recall scores of 1.00 ± 0.01, 
    indicating that they are able to correctly identify all positive instances. 
    However, it's important to note that the precision scores for these classifiers 
    are slightly lower compared to Logistic Regression and K-Nearest Neighbor. 
    But, based on the evaluation metrics for hate speech detection in NLP, 
    the best classifier can be determined by considering the F1-score, 
    which is a measure of the model's overall performance. By looking at the F1-scores, 
    Logistic Regression has the highest F1-score of 0.88 ± 0.02, followed closely by 
    Decision Tree, Random Forest, and SVM - SVC, all with F1-scores of 0.87 ± 0.03. 
    Therefore, based on the F1-scores, Logistic Regression appears to be the best 
    classifier for hate speech detection in NLP. In addition, Logistic Regression has 
    the highest precision score of 0.83 ± 0.04. It also has a relatively high recall.""")
    # Horizontal line separator
    st.markdown("---")
 
elif selected == "Try The Model":
    st.title("Try The Logistic Regression Model")
    st.write("This is the model page.")
    # Horizontal line separator
    st.markdown("---")
    
elif selected == "About":
    st.title("About")
    st.write("This is the about page.")
    # Horizontal line separator
    st.markdown("---")

elif selected == "Contact":
    st.title("Contact")
    st.write("This is the contact page.")
    # Horizontal line separator
    st.markdown("---")
