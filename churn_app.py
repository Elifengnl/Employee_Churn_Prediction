import streamlit as st
import pickle
import pandas as pd
from PIL import Image
import plotly.express as px



# Main title
st.sidebar.title('Employee Churn Prediction')
image = Image.open("images/office.jpg")
st.sidebar.image(image, use_column_width=True)
# Html adjustments to display the front end aspects
html_temp = """
<div style="background-color:darkblue;padding:10px">
<h2 style="color:white;text-align:center;">Employee Churn Prediction </h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

#st.image(image, use_column_width=True)


# Display the Dataset
#st.header("_HR Dataset_")
df = pd.read_csv('HR_Dataset.csv')
#st.write(df.head())
df1=df.copy()
df1=df1.drop('left', axis=1)

# Visualisation with Plotly
st.write('')
st.write('')
selected_column = st.selectbox("Select Column for X-axis", options=df1.columns )

department_left_counts = df.groupby([selected_column , 'left']).size().reset_index(name='count')
fig = px.bar(department_left_counts, x=selected_column, y='count', color='left', barmode='relative',
             title='Turnover Count by '+selected_column)

st.plotly_chart(fig, use_container_width=True)



# Side bar user inputs
department=st.sidebar.selectbox("Department", ('Sales', 'Technical', 'Support', 'IT', 'R&D', 'Product Management', 'Marketing', 'Accounting', 'HR', 'Management'))
salary=st.sidebar.radio('Salary',('Low','Medium','High'))
promotion=st.sidebar.radio("Promotion last 5 years:",('Yes','No'))
time=st.sidebar.slider("Length of service", 1, 10, step=1)
project=st.sidebar.slider("Number of projects", 1,10, step=1)
hours=st.sidebar.slider("Average monthly working hours", 80,310, step=10)
accident=st.sidebar.radio("Accident", ('Yes','No'))
satisfaction=st.sidebar.slider("Satisfaction score", 0.1,1.0, step=0.1)
evaluation=st.sidebar.slider("Last evaluation score", 0.1,1.0, step=0.1)

if promotion=='Yes':
    promotion=1
else :
    promotion=0     

if accident=='Yes':
    accident=1
else :
    accident=0      

# Converting user inputs to dataframe 
my_dict = {"satisfaction_level": satisfaction,
           "last_evaluation": evaluation,
           "number_project": project,
           'average_montly_hours': hours,
           'time_spend_company': time,
           "work_accident": accident,
           "promotion_last_5years": promotion,
           "departments": department,
           "salary": salary
}
df2 = pd.DataFrame.from_dict([my_dict])
df2.index = [''] * df2.shape[0]


st.header("You selected the following configuration:")
st.table(df2)


# Loading the model(s) to make predictions
loaded_model=pickle.load(open("XGB_model_with_transformer.pkl","rb"))
#transformer = pickle.load(open('transformer.pkl', 'rb'))
#df3 = transformer.transform(df2)

# defining the function which will make the prediction using the data
def get_prediction(model, input_data):
	prediction = model.predict(input_data)
	return prediction


st.subheader("Press the 'Predict' button below to get a prediction")

if st.button("Predict"):
    result = get_prediction(loaded_model, df2)[0]
    if result == 0:
        result2 = "stay"
        st.image(Image.open("images/lovemyjob.jpg"))
        st.success(f"The employee is likely to **{result2}**")
        
    elif result == 1:
        result2 = "leave"
        st.image(Image.open("images/quit.jpg"))
        st.success(f"The employee is likely to **{result2}**")
       
