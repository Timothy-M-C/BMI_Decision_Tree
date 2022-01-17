#create dataframe
df = pd.read_csv('/kaggle/input/bmidataset/bmi.csv')

#define label
label = df['Index']

#replace female/male with 0/1 which works better with scikitlearn
df["Gender"].replace({"Female": "0", "Male": "1"}, inplace=True)

#label independent variables, score is about .85 with gender included, .87 without gender.
datapoints = df[['Height','Weight']]

#create test and training data sets
training_datapoints, test_datapoints, training_labels, test_labels = train_test_split(datapoints, label, test_size = 0.3, random_state = 1)

#find what tree depths lead to what scores. It seems a score of 88 is the peak when depth is 10
scores = []
for i in range(1,21):
    classifier = DecisionTreeClassifier(random_state = 1, max_depth = i)
    classifier.fit(training_datapoints, training_labels)
    scores.append(classifier.score(test_datapoints,test_labels))
    print(classifier.score(test_datapoints,test_labels))
#visiualiztion of how scores change based on depth
plt.plot(range(1,21),scores)
#visual plot of the decision tree
#fig = plt.figure(figsize=(25,20))
#_ = tree.plot_tree(classifier,filled=True)

print(classifier.score(test_datapoints,test_labels))