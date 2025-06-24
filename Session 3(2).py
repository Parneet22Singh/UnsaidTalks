#Dice Game
import random

# Get number of players
num_players = int(input("Enter number of players: "))

# Get player names
players = []
for i in range(num_players):
    name = input(f"Enter name for player {i + 1}: ")
    players.append(name)

# Get number of rounds (minimum 10)
while True:
    total_rounds = int(input("Enter number of rounds (minimum 10): "))
    if total_rounds >= 10:
        break
    print("❗ Please enter at least 10 rounds.")

# Initialize scores and previous rolls
scores = {player: 0 for player in players}
previous_rolls = {player: None for player in players}

# Game rounds
round_num = 1
while round_num <= total_rounds:
    print(f"\n--- Round {round_num} ---")
    for player in players:
        roll = random.randint(1, 6)
        print(f"{player} rolls a {roll}")

        # Apply game rules
        if roll == 6:
            scores[player] += 10
            if previous_rolls[player] == 6:
                scores[player] += 5
                print(f"Power-Up! Two 6s in a row for {player} → +5 bonus points")
        elif roll == 1:
            scores[player] -= 5
        else:
            scores[player] += roll

        # Prevent negative score
        if scores[player] < 0:
            scores[player] = 0

        previous_rolls[player] = roll  # update last roll
        print(f"  Score for {player}: {scores[player]}")

    round_num += 1

# Show final results
print("\n=== Final Scores ===")
for player, score in scores.items():
    print(f"{player}: {score}")

# Determine winner
winner = max(scores.items(), key=lambda x: x[1])
print(f"\nWinner: {winner[0]} with {winner[1]} points!")
