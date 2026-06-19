import random

# Week 4 Syllabus: Genetic Algorithm Implementation
class TreatmentGeneticAlgorithm:
    def __init__(self, diagnosis, bmi, target_calories):
        self.diagnosis = diagnosis
        self.bmi = bmi
        self.target_calories = target_calories
        
        # Genes (Possible treatments)
        self.exercises = ["Cardio (30m)", "Yoga", "Weight Training", "HIIT", "Brisk Walk (45m)", "Rest"]
        self.meals = ["High Protein", "Low Carb", "Mediterranean", "Keto", "DASH Diet", "Vegetarian"]

    def generate_random_chromosome(self):
        # A chromosome is a 7-day plan (7 pairs of exercise + meal)
        return [{"exercise": random.choice(self.exercises), "meal": random.choice(self.meals)} for _ in range(7)]

    def calculate_fitness(self, chromosome):
        fitness = 0
        for day in chromosome:
            # If Diabetic, punish high carb and reward cardio
            if self.diagnosis == "diabetes":
                if day["meal"] == "Low Carb" or day["meal"] == "DASH Diet": fitness += 10
                if day["exercise"] == "Cardio (30m)": fitness += 5
            
            # If Heart Disease, reward DASH diet and light cardio
            elif self.diagnosis == "heart":
                if day["meal"] == "DASH Diet" or day["meal"] == "Mediterranean": fitness += 10
                if day["exercise"] == "HIIT": fitness -= 10 # HIIT is bad for heart patients!
                if day["exercise"] == "Brisk Walk (45m)": fitness += 5
                
            # If Anemia, reward High Protein
            elif self.diagnosis == "anemia":
                if day["meal"] == "High Protein": fitness += 10
                
            # Punish resting too much
            if day["exercise"] == "Rest": fitness -= 2
            
        return fitness

    def crossover(self, parent1, parent2):
        # Split the week in half and combine parents
        split = 3
        child = parent1[:split] + parent2[split:]
        return child

    def mutate(self, chromosome):
        # 10% chance to mutate a day's plan
        if random.random() < 0.1:
            mutate_day = random.randint(0, 6)
            chromosome[mutate_day]["exercise"] = random.choice(self.exercises)
            chromosome[mutate_day]["meal"] = random.choice(self.meals)
        return chromosome

    def generate_plan(self, generations=50, population_size=20):
        # 1. Initialize Population
        population = [self.generate_random_chromosome() for _ in range(population_size)]
        
        for _ in range(generations):
            # 2. Evaluate Fitness
            population = sorted(population, key=lambda x: self.calculate_fitness(x), reverse=True)
            
            # 3. Keep the best (Elitism)
            next_generation = population[:5] 
            
            # 4. Crossover & Mutate
            while len(next_generation) < population_size:
                p1 = random.choice(population[:10])
                p2 = random.choice(population[:10])
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                next_generation.append(child)
                
            population = next_generation
            
        # Return the absolute fittest plan
        best_plan = population[0]
        return best_plan