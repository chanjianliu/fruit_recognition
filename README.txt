This is a school project where a machine learning model is built to recognise photos of different fruits, namely apple, orange, banana, and mixed fruit photos. (the categories were encoded and represented by numbers in the application)

dataset containing the photos of fruits are included, separated by a folder containing data used for training and another used for testing.

the photos were first converted to all RGB, removing the transperency layer ('a') in photos of RGBa format.

data of the photos are then put into a numpy array and scaled to a range between 0 to 1 before going into training.

The model yields the result of an accuracy between 85% to 90%.