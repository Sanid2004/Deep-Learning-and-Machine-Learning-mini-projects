import pandas as pd
import numpy as np

# Data collection

user_data = {
    'user': [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4],
    'features': [
        "Horror", "Drama", "Action", "IMdB Rating",
        "Horror", "Action", "IMdB Rating", 
        "Drama", "Action", "Horror",
        "Horror", "Drama", "Action", "IMdB Rating"
    ],
    'values': [
        1,0,1,1,
        1,0,1,
        1,0,0,
        1,1,1,0
    ]
}

product_data = {
    'features' : ["Horror", "Horror","Horror","Horror","Drama", "Drama","Drama" , "Drama" , "Action", "Action","Action","Action","IMdB Rating","IMdB Rating","IMdB Rating","IMdB Rating"],
    'movies' : ["3 Idiots" , "Anabelle" , "Krish 3" , "12th fail",
                "3 Idiots" , "Anabelle" , "Krish 3" , "12th fail",
                "3 Idiots" , "Anabelle" , "Krish 3" , "12th fail",
                "3 Idiots" , "Anabelle" , "Krish 3" , "12th fail"],
    'values' : [0,9,2,1,9,3,7,6,6,4,9,4,9,8,7,10],
}

df = pd.DataFrame(user_data) 
matrix = df.pivot(index='user' , columns='features' , values='values')
print(matrix)

df2 = pd.DataFrame(product_data)
matrix2 = df2.pivot(index='features',columns='movies',values='values')
print(matrix2)

# Sparsity 

# Calculate sparsity for the user matrix
total_elements_user_matrix = matrix.size
zero_elements_user_matrix = matrix.isna().sum().sum()  # null  entries
sparsity_user_matrix =  (zero_elements_user_matrix / total_elements_user_matrix)

# Calculate sparsity for the product matrix
total_elements_product_matrix = matrix2.size
zero_elements_product_matrix = matrix2.isna().sum().sum()  # null entries
sparsity_product_matrix = (zero_elements_product_matrix / total_elements_product_matrix)

print(f"Sparsity of User Matrix: {sparsity_user_matrix:.2f}")
print(f"Sparsity of Product Matrix: {sparsity_product_matrix:.2f}")

# Matrix factorization
matrix.fillna(0,inplace=True)
matrix2.fillna(0,inplace=True)
matrix_np = matrix.to_numpy()
matrix2_np = matrix2.to_numpy()
result = np.dot(matrix_np, matrix2_np)
result_df = pd.DataFrame(result, index=matrix.index, columns=matrix2.columns)
print("Matrix Multiplication Result:")
print(result_df)


# Recommendation
# Get the top recommended movies for each user
# Calculate the maximum value from each row
max_values_per_row = result_df.max(axis=1)  # Maximum value
max_columns_per_row = result_df.idxmax(axis=1)  # Column name for the maximum value

# Combine results into a DataFrame for clarity
max_info = pd.DataFrame({
    'Max Value': max_values_per_row,
    'Movie': max_columns_per_row
})

print("Maximum values and movies for each user:")
print(max_info)

# Model Evaluation
# For this example, we'll use a simple evaluation metric: the number of recommended movies that are
# actually watched by the user 

user_watched_movie = {
    1:'Anabelle',
    2:'Anabelle',
    3:'3 Idiots',
    4:'Krish 3'
}
max_info['User Watched'] = [user_watched_movie.get(i + 1) for i in range(len(max_info))]
max_info['Match'] = max_info['Movie'] == max_info['User Watched']

print(max_info)

# Calculate accuracy
accuracy = max_info['Match'].mean()
print(f"\nRecommendation Accuracy: {accuracy:.2%}")

def recommend(search):
  search_list = search.split()
  categories=['Action',  'Drama'  ,'Horror'  ,'IMdBRating']
  l1=[]
  for i in categories:
    if i in search_list:
      l1.append(1)
    else:
      l1.append(0)
  matrix_fac = np.dot(l1,matrix2)
  l3 = list(matrix2.columns)
  max_value = max(matrix_fac)
  max_index = np.where(matrix_fac == max_value)[0][0]
  return l3[max_index]

res = recommend("Action IMdBRating")
print(res)
