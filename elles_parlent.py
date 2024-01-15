import pandas as pd
import numpy as np  

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('elles_parlent_dataset.csv')

# Use the drop method to remove the 'Jeune' column
df.drop(columns=['Unnamed: 0'], inplace=True)

# Define a dictionary to map old column names to new column names
column_mapping = {
    'Unnamed: 1': 'Email',
    'Unnamed: 2': 'Nom',
    'Unnamed: 3': 'Age',
    'Unnamed: 4': 'Ville',
    'Unnamed: 5': 'Telephone',
    'Unnamed: 6': 'Niveau_Etude',
    'Unnamed: 7': 'Autres_Etude',
    'Unnamed: 8': 'Profession',
    'Unnamed: 9': 'Presentation_Personnelle',
    'Unnamed: 10': 'Presentation_Projet',
    'Unnamed: 11': 'Ambition',
    'Unnamed: 12': 'Plus_Belle_Reussite',
    'Unnamed: 13': 'Qualites',
    'Unnamed: 14': 'Faiblesses',
    'Unnamed: 15': 'Interet',
    'Unnamed: 16': 'Information_Concours',
    'Unnamed: 17': 'Autres',
    'Unnamed: 18': 'Participation_Concours',
    'Unnamed: 19': 'Autre_Concours',
    'Unnamed: 20': 'Langue',
    
}

# Use the rename method to rename columns
df.rename(columns=column_mapping, inplace=True)


# Use the drop method to remove the 'Jeune' column
df.drop(columns=['Jeune'], inplace=True)

# Use the drop method to remove the first row
df.drop(index=0, inplace=True)

# Convert 'Age' column to integer
df['Age'] = pd.to_numeric(df['Age'], errors='coerce').astype('Int64')

# Drop rows with NaN values in the 'Age' column
df = df.dropna(subset=['Age'])

# Convert 'Telephone' column to integer
df['Telephone'] = pd.to_numeric(df['Telephone'], errors='coerce').astype('Int64')
print(df['Age'])

# Define a function to categorize age
def categorize_age(age):
    if age < 21:
        return 'Minime'
    elif 21 <= age <= 25:
        return 'Jeune'
    else:
        return 'Senior'

# Apply the function to create the 'Categorie' column
df['Categorie'] = df['Age'].apply(categorize_age)
print(df['Categorie'])

# Display the counts of each category in the 'Categorie' column
categorie_counts = df['Categorie'].value_counts()

# Display the statistics
print("Statistics for the 'Categorie' column:")
print(categorie_counts)

""" 
Statistics for the 'Categorie' column:
Categorie
Jeune     29
Senior    22
Minime    13
"""

# Display the counts of each category in the 'Categorie' column
ville_counts = df['Ville'].value_counts()

# Display the statistics
print("Statistics for the 'Ville' column:")
print(ville_counts)


""" 
Statistics for the 'Ville' column:
Ville
Yaounde    46
Douala     18
Name: count, dtype: int64
"""

# Display the counts of each unique combination of 'Ville' and 'Categorie'
combined_stats = df.groupby(['Ville', 'Categorie']).size().reset_index(name='Counts')

# Display the statistics
print("Statistics for the combination of 'Ville' and 'Categorie':")
print(combined_stats)

""" 
Statistics for the combination of 'Ville' and 'Categorie':
     Ville Categorie  Counts
0   Douala     Jeune       8
1   Douala    Minime       2
2   Douala    Senior       8
3  Yaounde     Jeune      21
4  Yaounde    Minime      11
5  Yaounde    Senior      14
"""
# Create a bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x='Ville', y='Counts', hue='Categorie', data=combined_stats)
plt.title('Total des combinaisons de Ville et Categorie')
plt.show()

# Display the counts of each unique combination of 'Nom', 'Ville', and 'Categorie'
combined_stats = df.groupby(['Nom', 'Ville', 'Categorie']).size().reset_index(name='Counts')

# Display the statistics
print("Statistics for the combination of 'Nom', 'Ville' and 'Categorie':")
print(combined_stats)

""" 
Statistics for the combination of 'Ville' and 'Categorie':
     Ville Categorie  Counts
0   Douala     Jeune       8
1   Douala    Minime       2
2   Douala    Senior       8
3  Yaounde     Jeune      21
4  Yaounde    Minime      11
5  Yaounde    Senior      14
"""


# Display the counts of each unique combination of 'Ville' and 'Langue'
counts_by_ville_langue = df.groupby(['Ville', 'Langue']).size().reset_index(name='Counts')

# Display the result
print(counts_by_ville_langue)

# Create a bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x='Ville', y='Counts', hue='Langue', data=counts_by_ville_langue)
plt.title('Counts of speakers by Ville and Langue')
plt.show()

# Assuming df is your DataFrame
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=20)
plt.title('Age Distribution of Speakers')
plt.xlabel('Age')
plt.show()

from wordcloud import WordCloud

# Analyse des qualités
qualities_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['Qualites'].dropna()))
plt.figure(figsize=(10, 5))
plt.imshow(qualities_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nuage de qualités des candidates')
plt.show()

# Assuming df is your DataFrame
qualities_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['Faiblesses'].dropna()))
plt.figure(figsize=(10, 5))
plt.imshow(qualities_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nuage de faiblesses des candidates')
plt.show()
print(df.head())
print(df.info())

import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load a world map for plotting
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Assuming df is your DataFrame
speaker_locations = df.groupby('Ville').size().reset_index(name='SpeakerCount')
merged = world.merge(speaker_locations, how='left', left_on='name', right_on='Ville')
merged = merged.fillna(0)

# Plot the map
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
merged.plot(column='SpeakerCount', ax=ax, legend=True, cax=cax, legend_kwds={'label': "Number of Speakers"})
plt.title('Geographical Distribution of Speakers')
#plt.show()

# Map 'oui' to True and 'non' to False
df['Participation_Concours_Bool'] = df['Participation_Concours'].map({'Oui': True, 'Non': False})

# Display the counts of True and False values
participation_counts = df['Participation_Concours_Bool'].value_counts()
print("Counts of Participation in Contests (True/False):")
print(participation_counts)

# Assuming df is your DataFrame
qualities_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['Interet'].dropna()))
plt.figure(figsize=(10, 5))
plt.imshow(qualities_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nuage Interêt des candidates')
plt.show()
print(df.head())
print(df.info())

# Display the counts of each unique value in 'Niveau_Etude'
niveau_etude_counts = df['Niveau_Etude'].value_counts()
print("Counts of Niveau_Etude:")
print(niveau_etude_counts)

# Create a bar chart
plt.figure(figsize=(12, 6))
sns.barplot(x=niveau_etude_counts.index, y=niveau_etude_counts.values)
plt.title('Counts of Niveau_Etude')
plt.xlabel('Niveau_Etude')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.show()


# Display the top N counts of 'Niveau_Etude'
top_n = 10  # Adjust the value of N as needed
top_niveau_etude_counts = df['Niveau_Etude'].value_counts().nlargest(top_n)
print("Top {} Niveau_Etude counts:".format(top_n))
print(top_niveau_etude_counts)

# Create a bar chart for the top N values
plt.figure(figsize=(12, 6))
sns.barplot(x=top_niveau_etude_counts.index, y=top_niveau_etude_counts.values)
plt.title('Top {} Niveau_Etude counts'.format(top_n))
plt.xlabel('Niveau_Etude')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.show()
