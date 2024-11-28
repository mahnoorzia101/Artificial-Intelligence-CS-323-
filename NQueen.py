import tkinter as tk
from tkinter import messagebox
import random


class NQueenSolver:
    def __init__(self, N):
        self.N = N
        self.population_size = 100
        self.mutation_rate = 0.1
        self.generations = 5000
        self.population = self.generate_initial_population()
        self.solutions = []  # List to store solutions and their generation
        self.find_all_solutions()

    def generate_initial_population(self):
        return [random.sample(range(self.N), self.N) for _ in range(self.population_size)]

    def fitness(self, chromosome):
        non_attacking = 0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if abs(chromosome[i] - chromosome[j]) != abs(i - j):  # Check diagonal attack
                    non_attacking += 1
        return non_attacking

    def selection(self):
        weighted_population = [(self.fitness(ch), ch) for ch in self.population]
        weighted_population.sort(reverse=True, key=lambda x: x[0])
        return [ch for _, ch in weighted_population[:self.population_size // 2]]

    def crossover(self, parent1, parent2):
        split = random.randint(1, self.N - 2)
        child = parent1[:split] + [g for g in parent2 if g not in parent1[:split]]
        return child

    def mutate(self, chromosome):
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(self.N), 2)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        return chromosome

    def evolve(self):
        new_population = []
        selected = self.selection()
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(selected, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        self.population = new_population

    def find_all_solutions(self):
        seen = set()
        for generation in range(self.generations):
            for chromosome in self.population:
                if self.fitness(chromosome) == self.N * (self.N - 1) // 2:  # Max fitness
                    solution = tuple(chromosome)
                    if solution not in seen:
                        self.solutions.append((solution, generation)) 
                        seen.add(solution)
            self.evolve()


class NQueenUI:
    def __init__(self, root):
        self.root = root
        self.root.title("N-Queens Problem Solver")

        # Input frame
        self.input_frame = tk.Frame(root)
        self.input_frame.pack()

        tk.Label(self.input_frame, text="Enter the value of N: ").grid(row=0, column=0)
        self.n_input = tk.Entry(self.input_frame, width=5)
        self.n_input.grid(row=0, column=1)
        tk.Button(self.input_frame, text="Start", command=self.start_solver).grid(row=0, column=2)

        # Board frame
        self.board_frame = tk.Frame(root)
        self.board_frame.pack()

        self.board = tk.Canvas(self.board_frame, width=500, height=500, bg="white")
        self.board.pack()

        # Navigation buttons
        self.navigation_frame = tk.Frame(root)
        self.navigation_frame.pack()

        self.next_button = tk.Button(self.navigation_frame, text="Next Solution", command=self.show_next_solution)
        self.next_button.pack(side="left")
        self.next_button.config(state="disabled")

        self.quit_button = tk.Button(self.navigation_frame, text="Quit", command=root.destroy)
        self.quit_button.pack(side="right")

        # Generation display
        self.generation_label = tk.Label(root, text="Generation: 0")
        self.generation_label.pack()

        self.solver = None
        self.current_solution_index = 0

    def start_solver(self):
        try:
            N = int(self.n_input.get())
            if N < 4:
                messagebox.showerror("Error", "N must be 4 or greater.")
                return
            self.solver = NQueenSolver(N)
            self.current_solution_index = 0

            if not self.solver.solutions:
                messagebox.showerror("No Solutions", f"No solutions found for N={N}.")
                self.next_button.config(state="disabled")
            else:
                self.next_button.config(state="normal")
                self.show_solution()

        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer for N.")

    def draw_board(self, solution):
        self.board.delete("all")
        N = len(solution)
        cell_size = 500 // N

        for i in range(N):
            for j in range(N):
                x1, y1 = i * cell_size, j * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size
                color = "white" if (i + j) % 2 == 0 else "gray"
                self.board.create_rectangle(x1, y1, x2, y2, fill=color)

        for row, col in enumerate(solution):
            x1, y1 = col * cell_size, row * cell_size
            x2, y2 = x1 + cell_size, y1 + cell_size
            self.board.create_oval(x1 + 5, y1 + 5, x2 - 5, y2 - 5, fill="red")

    def show_solution(self):
        if self.current_solution_index < len(self.solver.solutions):
            solution, generation = self.solver.solutions[self.current_solution_index]
            self.draw_board(solution)
            self.generation_label.config(text=f"Generation: {generation}")
        else:
            self.board.delete("all")
            self.board.create_text(250, 250, text="No More Solutions", font=("Arial", 24), fill="black")

    def show_next_solution(self):
        self.current_solution_index += 1
        self.show_solution()


if __name__ == "__main__":
    root = tk.Tk()
    app = NQueenUI(root)
    root.mainloop()
