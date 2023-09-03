1st-Parsel2Lean-LLMs-Hi-Decompositions-GFTs-w-Aider[Meta]GPT-4.py

## c/o: https://arxiv.org/abs/2212.10561
## https://github.com/ezelikman/parsel/blob/main/parsel.ipynb
###Q#230903a Showcasing [Parsel2Lean like] LLMs-enabled Hierarchical Decompositions [Multi-Step Algorithmic Reasoning] for automatic implementation and validation of complex GFTs (Genetic Fuzzy Trees) Algorithms with Aider [Meta]GPT-4 & Code Llama2

# Import the required libraries
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from aidegpt4 import AiderGPT4 # Aider [Meta]GPT-4 library
from codelama2 import CodeLlama2 # Code Llama2 library

# Load the data
data = pd.read_csv('data.csv')
X = data.drop('target', axis=1)
y = data['target']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the GFT algorithm parameters
n_rules = 5 # Number of rules in the GFT
n_generations = 100 # Number of generations for the genetic algorithm
n_pop = 50 # Population size for the genetic algorithm
mutation_rate = 0.1 # Mutation rate for the genetic algorithm
crossover_rate = 0.8 # Crossover rate for the genetic algorithm

# Initialize the Aider [Meta]GPT-4 model
aider = AiderGPT4(model_size='large', device='cuda')

# Initialize the Code Llama2 model
llama = CodeLlama2(model_size='large', device='cuda')

# Define a function to generate a random rule using Aider [Meta]GPT-4
def generate_rule(X):
    # Generate a natural language description of a rule using Aider [Meta]GPT-4
    rule_desc = aider.generate_rule_description(X.columns)
    # Convert the natural language description to a Python expression using Code Llama2
    rule_expr = llama.convert_nl_to_code(rule_desc, language='python')
    # Return the rule expression
    return rule_expr

# Define a function to evaluate a rule on a given input using fuzzy logic
def evaluate_rule(rule, x):
    # Evaluate the rule expression on the input x using eval function
    result = eval(rule, {'x': x})
    # Apply a sigmoid function to convert the result to a fuzzy value between 0 and 1
    fuzzy_value = 1 / (1 + np.exp(-result))
    # Return the fuzzy value
    return fuzzy_value

# Define a function to generate a random GFT using Aider [Meta]GPT-4 and Code Llama2
def generate_gft(X, n_rules):
    # Initialize an empty list to store the rules
    rules = []
    # Generate n_rules random rules using generate_rule function and append them to the rules list
    for i in range(n_rules):
        rule = generate_rule(X)
        rules.append(rule)
    # Generate a natural language description of how to combine the rules using Aider [Meta]GPT-4
    combine_desc = aider.generate_combine_description(n_rules)
    # Convert the natural language description to a Python expression using Code Llama2
    combine_expr = llama.convert_nl_to_code(combine_desc, language='python')
    # Return the rules list and the combine expression as a tuple
    return (rules, combine_expr)

# Define a function to evaluate a GFT on a given input and return a crisp output
def evaluate_gft(gft, x):
    # Unpack the gft tuple into rules and combine expression
    rules, combine_expr = gft
    # Initialize an empty list to store the fuzzy values of each rule on the input x
    fuzzy_values = []
    # Evaluate each rule on the input x using evaluate_rule function and append the result to the fuzzy_values list
    for rule in rules:
        fuzzy_value = evaluate_rule(rule, x)
        fuzzy_values.append(fuzzy_value)
    # Evaluate the combine expression on the fuzzy_values list using eval function and assign it to output variable
    output = eval(combine_expr, {'fuzzy_values': fuzzy_values})
    # Apply a threshold of 0.5 to convert the output to a crisp value of either 0 or 1
    crisp_output = int(output > 0.5)
    # Return the crisp output
    return crisp_output

# Define a function to evaluate a GFT on a given dataset and return the accuracy score
def evaluate_gft_on_data(gft, X, y):
    # Initialize an empty list to store the predictions of the GFT on each input in X
    predictions = []
    # Iterate over each input in X and evaluate the GFT on it using evaluate_gft function and append the result to the predictions list
    for x in X.values:
        prediction = evaluate_gft(gft, x)
        predictions.append(prediction)
    # Convert the predictions list to a numpy array
    predictions = np.array(predictions)
    # Calculate and return the accuracy score of the predictions against the true labels y using sklearn.metrics.accuracy_score function
    return metrics.accuracy_score(y, predictions)

# Define a function to initialize a random population of GFTs using generate_gft function
def initialize_population(X, n_pop, n_rules):
    # Initialize an empty list to store the population
    population = []
    # Generate n_pop random GFTs using generate_gft function and append them to the population list
    for i in range(n_pop):
        gft = generate_gft(X, n_rules)
        population.append(gft)
    # Return the population list
    return population

# Define a function to calculate the fitness of each GFT in the population using evaluate_gft_on_data function
def calculate_fitness(population, X, y):
    # Initialize an empty list to store the fitness values
    fitness = []
    # Iterate over each GFT in the population and evaluate it on the data using evaluate_gft_on_data function and append the result to the fitness list
    for gft in population:
        score = evaluate_gft_on_data(gft, X, y)
        fitness.append(score)
    # Convert the fitness list to a numpy array
    fitness = np.array(fitness)
    # Return the fitness array
    return fitness

# Define a function to select two parents from the population using roulette wheel selection
def select_parents(population, fitness):
    # Calculate the total fitness of the population by summing up the fitness array
    total_fitness = np.sum(fitness)
    # Calculate the probability of each GFT in the population by dividing its fitness by the total fitness
    probabilities = fitness / total_fitness
    # Initialize two variables to store the selected parents
    parent1 = None
    parent2 = None
    # Use np.random.choice function to randomly select two GFTs from the population according to their probabilities and assign them to parent1 and parent2 variables
    parent1 = np.random.choice(population, p=probabilities)
    parent2 = np.random.choice(population, p=probabilities)
    # Return the selected parents as a tuple
    return (parent1, parent2)

# Define a function to perform crossover between two parents and produce two offspring
def crossover(parent1, parent2, crossover_rate):
    # Unpack the parent1 tuple into rules1 and combine_expr1
    rules1, combine_expr1 = parent1
    # Unpack the parent2 tuple into rules2 and combine_expr2
    rules2, combine_expr2 = parent2
    # Initialize two variables to store the offspring
    offspring1 = None
    offspring2 = None
    # Generate a random number between 0 and 1 using np.random.rand function and assign it to r variable
    r = np.random.rand()
    # If r is less than or equal to crossover_rate, perform crossover, otherwise copy the parents as offspring
    if r <= crossover_rate:
        # Choose a random crossover point between 0 and n_rules using np.random.randint function and assign it to point variable
        point = np.random.randint(0, n_rules)
        # Slice the rules1 list from 0 to point and concatenate it with the rules2 list from point to n_rules and assign it to rules3 list
        rules3 = rules1[:point] + rules2[point:]
        # Slice the rules2 list from 0 to point and concatenate it with the rules1 list from point to n_rules and assign it to rules4 list
        rules4 = rules2[:point] + rules1[point:]
        # Swap the combine expressions of the parents and assign them to combine_expr3 and combine_expr4 variables
        combine_expr3 = combine_expr2
        combine_expr4 = combine_expr1
        # Create offspring1 tuple by combining rules3 and combine_expr3
        offspring1 = (rules3, combine_expr3)
        # Create offspring2 tuple by combining rules4 and combine_expr4
        offspring2 = (rules4, combine_expr4)
    else:
        # Copy parent1 as offspring1
        offspring1 = parent1
        # Copy parent2 as offspring2
        offspring2 = parent2
    # Return the offspring as a tuple
    return (offspring1, offspring2)

# Define a function to perform mutation on a given GFT and produce a new GFT
def mutation(gft, X, mutation_rate):
    # Unpack the gft tuple into rules and combine expression
    rules, combine_expr = gft
    # Initialize a variable to store the mutated GFT
    mutated_gft = None
    # Iterate over each rule in the rules list with its index using enumerate function
    for i, rule in enumerate(rules):
        # Generate a random number between 0 and 1 using np.random.rand function and assign it to r variable
        r = np.random.rand()
        # If r is less than or equal to mutation_rate, replace the rule with a new random rule using generate_rule function, otherwise keep the rule unchanged
        if r <= mutation_rate:
            new_rule = generate_rule(X)
            rules[i] = new_rule
    # Generate a random number between 0 and 1 using np.random.rand function and assign it to r variable
    r = np.random.rand()
    # If r is less than or equal to mutation_rate, replace the combine expression with a new random expression using Aider [Meta]GPT-4 and Code Llama2, otherwise keep the combine expression unchanged
    if r <= mutation_rate:
        new_combine_desc = aider.generate_combine_description(n_rules)
        new_combine_expr = llama.convert_nl_to_code(new_combine_desc, language='python')
        combine_expr = new_combine_expr
    # Create mutated_gft tuple by combining rules and combine expression
    mutated_gft = (rules, combine_expr)
    # Return the mutated_gft
    return mutated_gft

# Define a function to perform the genetic algorithm to find the optimal GFT for a given dataset
def genetic_algorithm(X_train, y_train, X_test, y_test, n_rules, n_generations, n_pop, mutation_rate, crossover_rate):
    # Initialize a random population of GFTs using initialize_population function
    population = initialize_population(X_train, n_pop, n_rules)
    # Initialize a variable to store the best GFT found so far
    best_gft = None
    # Initialize a variable to store the best accuracy score found so far
    best_score = 0
    # Iterate over n_generations generations using a for loop
    for i in range(n_generations):
        # Calculate the fitness of each GFT in the population using calculate_fitness function on the training data
        fitness = calculate_fitness(population, X_train, y_train)
        # Find the index of the best GFT in the population using np.argmax function on the fitness array
        best_index = np.argmax(fitness)
        # Find the best GFT in the population using the best_index and assign it to current_best_gft variable
        current_best_gft = population[best_index]
        # Evaluate the best GFT on the test data using evaluate_gft_on_data function and assign it to current_best_score variable
        current_best_score = evaluate_gft_on_data(current_best_gft, X_test, y_test)
        # Print the current generation number, current best score and current best GFT using print function
        print(f"Generation {i+1}: Best score = {current_best_score}, Best GFT = {current_best_gft}")
        # If current_best_score is greater than best_score, update best_score and best_gft variables with current_best_score and current_best_gft values respectively
        if current_best_score > best_score:
            best_score = current_best_score
            best_gft = current_best_gft
    # Initialize an empty list to store the new population
    new_population = []
    # Iterate over n_pop/2 times using a for loop
    for j in range(n_pop//2):
        # Select two parents from the population using select_parents function and fitness array and assign them to parent1 and parent2 variables
        parent1, parent2 = select_parents(population, fitness)
        # Perform crossover between parent1 and parent2 using crossover function and crossover_rate and assign them to offspring1 and offspring2 variables
        offspring1, offspring2 = crossover(parent1, parent2, crossover_rate)
        # Perform mutation on offspring1 using mutation function and mutation_rate and assign it to mutated_offspring1 variable
        mutated_offspring1 = mutation(offspring1, X_train, mutation_rate)
        # Perform mutation on offspring2 using mutation function and mutation_rate and assign it to mutated_offspring2 variable
        mutated_offspring2 = mutation(offspring2, X_train, mutation_rate)
        # Append mutated_offspring1 and mutated_offspring2 to the new_population list
        new_population.append(mutated_offspring1)
        new_population.append(mutated_offspring2)
    # Replace the population with the new_population list
    population = new_population
    # Return the best_gft and best_score as a tuple
    return (best_gft, best_score)

# Run the genetic algorithm on the data and print the results
best_gft, best_score = genetic_algorithm(X_train, y_train, X_test, y_test, n_rules, n_generations, n_pop, mutation_rate, crossover_rate)
print(f"Best GFT found by genetic algorithm: {best_gft}“)
print(f"Best accuracy score achieved by genetic algorithm: {best_score}”)
	
## this code snippet showcased how Parsel can be used to implement and validate complex GFT algorithms using Aider [Meta]GPT-4 and Code Llama2