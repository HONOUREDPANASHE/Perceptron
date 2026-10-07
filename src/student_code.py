import common


def part_one_classifier(data_train, data_test):
    
    # Initialisation
    weights = [0.0, 0.0, 0.0]
    learning_rate = 0.1
    max_epochs = 1000
    
    # Train 
    for epoch in range(max_epochs):
        
        errors = 0
        
        for i in range(common.constants.TRAINING_SIZE):
            
            x = data_train[i][0]
            
            y = data_train[i][1]
            label = data_train[i][2]
            
            activation = weights[0] * x + weights[1] * y + weights[2]
            
            # Predict: if activation >= 0, predict 1, else predict 0
            prediction = 1 if activation >= 0 else 0
            
            # Update weights if prediction is wrong
            if prediction != label:
                errors += 1
                # target - prediction gives us the error direction
                error = label - prediction
                weights[0] += learning_rate * error * x
                weights[1] += learning_rate * error * y
                weights[2] += learning_rate * error  # bias update
        
        # If no errors, we've converged
        if errors == 0:
            break
    
    # Classify test data
    for i in range(common.constants.TEST_SIZE):
        x = data_test[i][0]
        y = data_test[i][1]
        
        activation = weights[0] * x + weights[1] * y + weights[2]
        prediction = 1 if activation >= 0 else 0
        
        data_test[i][2] = prediction


def part_two_classifier(data_train, data_test):
   
    num_c = 10
   
    weights = [[0.0, 0.0] for _ in range(num_c)]
    learning_rate = 0.01
    max_epochs = 1000
    
    # Train
    for epoch in range(max_epochs):
        errors = 0
        for i in range(common.constants.TRAINING_SIZE):
            x = data_train[i][0]
            y = data_train[i][1]
            true_label = int(data_train[i][2])
            
            # Compute activation for each class
            activations = []
            for c in range(num_c):
                activation = weights[c][0] * x + weights[c][1] * y
                activations.append(activation)
            
            # Predict the class with highest activation	
            predicted_label = activations.index(max(activations))
            
            

            if predicted_label != true_label:
                errors += 1
                # Increase weights for correct class
                weights[true_label][0] += learning_rate * x
                weights[true_label][1] += learning_rate * y
                
                # Decrease weights for incorrectly predicted class
                weights[predicted_label][0] -= learning_rate * x
                weights[predicted_label][1] -= learning_rate * y
        
        # Early stopping if converged
        if errors == 0:
            break
    
    # Classify test data
    for i in range(common.constants.TEST_SIZE):
        x = data_test[i][0]
        y = data_test[i][1]
        
        # Compute activation for each class
        activations = []
        for c in range(num_c):
            activation = weights[c][0] * x + weights[c][1] * y
            activations.append(activation)
        
    
        predicted_label = activations.index(max(activations))
        data_test[i][2] = predicted_label