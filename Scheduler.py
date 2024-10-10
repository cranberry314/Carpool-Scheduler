#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 11:01:03 2024
Carpool Scheduler

The purpose of this script is to schedule carpools so that parents can relax for 15 minutes.
Y = Yes! I can get the kids
N = No. I will be busy and cannot get them
M = Maybe, I am free but would prefer not

The script uses a genetic algorithm to build out the "best" schedule and returns the top 3. In order to use
google sheets you must enable the Service API and give the service email, found in the json file,
Editor access to the sheet.

The format of the Google Sheet is
Date | Name 1 | Name 2 | Name 3 | ...
1/1/2024 | Y | N | M
1/2/2024 | M | N | Y
etc

# Scheduler is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Budget Software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ExampleProject. If not, see
# <https://www.gnu.org/licenses/>.


@author: andrewfinn
"""
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import os
import numpy as np
import random

#os.getcwd()
os.chdir('/Users/Path/To/Carpool Scheduler Folder')

# Set up the credentials
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("carpool-scheduler.json", scope)
client = gspread.authorize(creds)

# Open the Google Sheet
sheet = client.open("Carpool Scheduler").sheet1

# Extract all data from the sheet
data = sheet.get_all_records()

# Convert to pandas DataFrame for easy manipulation
availability_df = pd.DataFrame(data)

# Convert 'Y', 'M', 'N' to numerical values: Y = 1, M = 0, N = -1
conversion_map = {'Y': 1, 'M': 0, 'N': -1}
df_converted = availability_df.replace(conversion_map)

# Define parameters
population_size = 100
generations = 1000 #500
mutation_rate = 0.3 #0.1
selection_rate = 0.2

# Fitness function: sum of availability values (Y=1, M=0) while respecting constraints (no N's allowed)
# Penalizes drivers who frequently mark 'M'
def fitness(schedule, df, balance_weight=0.5, penalize_non_drivers=True):
    score = 0
    drive_counts = {driver: 0 for driver in df.columns[1:]}  # Initialize drive counts
    penalty = 0  # Penalty for those marking 'M' without driving

    for i, driver in enumerate(schedule):
        if df.loc[i, driver] != -1:  # Check if driver is available
            score += df.loc[i, driver]  # Add availability score (1 for Y, 0 for M)
            drive_counts[driver] += 1  # Increment drive count for the driver

    # Calculate balancing term
    avg_drives = np.mean(list(drive_counts.values()))
    balancing_term = sum(abs(count - avg_drives) for count in drive_counts.values())

    # Penalty for individuals who mark 'M' frequently but don't drive
    if penalize_non_drivers:
        for driver in df.columns[1:]:
            # Number of 'M' (0) entries for this driver
            maybe_count = len(df[df[driver] == 0])
            # Penalize if driver marks 'M' but hasn't driven as much
            if maybe_count > 0 and drive_counts[driver] < maybe_count:
                penalty += (maybe_count - drive_counts[driver])

    # Adjust score with balancing term and penalty
    adjusted_score = score - balance_weight * balancing_term - penalty
    return adjusted_score

# Function to check if there are dates with all 'N'
def check_unavailable_dates(df):
    unavailable_dates = []
    for i in range(len(df)):
        if all(df.iloc[i, 1:] == -1):  # If all drivers for that date are unavailable
            unavailable_dates.append(df.iloc[i, 0])  # Append the date (first column)
    return unavailable_dates


# Generate an initial random population
def create_population(df, size):
    population = []
    for _ in range(size):
        schedule = []
        for i in range(len(df)):
            # Choose only drivers who are not 'N' (-1) for that date
            available_drivers = df.columns[1:][df.iloc[i, 1:] != -1]  # Skip the Date column
            if len(available_drivers) > 0:
                chosen_driver = random.choice(available_drivers)
                schedule.append(chosen_driver)
        population.append(schedule)
    return population

# Selection: Pick a portion of the best individuals
def selection(population, df, balance_weight):
    fitness_scores = [(individual, fitness(individual, df, balance_weight)) for individual in population]
    fitness_scores = sorted(fitness_scores, key=lambda x: x[1], reverse=True)
    top_individuals = [x[0] for x in fitness_scores[:int(selection_rate * len(fitness_scores))]]
    return top_individuals

# Crossover: Create offspring by combining two parents
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 2)  # Choose a crossover point
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Mutation function: Randomly alter a gene (i.e., reassign a driver for a day)
def mutate(individual, df):
    if random.random() < mutation_rate:
        day_index = random.randint(0, len(individual) - 1)
        available_drivers = df.columns[1:][df.iloc[day_index, 1:] != -1]  # Ensure driver is not N
        if len(available_drivers) > 0:
            individual[day_index] = random.choice(available_drivers)
    return individual


# Genetic Algorithm with balancing and penalty for 'M' without driving
def genetic_algorithm(df, population_size, generations, balance_weight=0.5, penalize_non_drivers=True):
    unavailable_dates = check_unavailable_dates(df)
    
    if unavailable_dates:
        print(f"Error: No one is available to drive on the following dates: {unavailable_dates}")
        return None

    population = create_population(df, size=population_size)

    for generation in range(generations):
        # Selection
        selected_individuals = selection(population, df, balance_weight)

        # Crossover and produce next generation
        next_generation = []
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(selected_individuals, 2)
            child1, child2 = crossover(parent1, parent2)
            next_generation.append(mutate(child1, df))
            next_generation.append(mutate(child2, df))

        population = next_generation

        # Print the best schedule every 100 generations
        if generation % 100 == 0:
            best_schedule = max(population, key=lambda x: fitness(x, df, balance_weight, penalize_non_drivers))
            print(f"Generation {generation}, Best Schedule: {best_schedule}, Fitness: {fitness(best_schedule, df, balance_weight, penalize_non_drivers)}")

    # Sort the final population by fitness and return the top 3 schedules
    population_fitness = [(individual, fitness(individual, df, balance_weight, penalize_non_drivers)) for individual in population]
    sorted_population = sorted(population_fitness, key=lambda x: x[1], reverse=True)
    
    top_3_schedules = sorted_population[:3]  # Get the top 3 schedules
    return top_3_schedules

# Run the genetic algorithm with balancing
top_3_schedules = genetic_algorithm(df_converted, population_size, generations, balance_weight=0.5)

# Output the top 3 balanced schedules
if top_3_schedules:
    print("Top 3 Balanced Schedules:")
    for rank, (schedule, score) in enumerate(top_3_schedules, 1):
        print(f"\nRank {rank} Schedule (Fitness: {score}):")
        for i, date in enumerate(availability_df['Date']):
            print(f"On {date}, {schedule[i]} will drive.")
