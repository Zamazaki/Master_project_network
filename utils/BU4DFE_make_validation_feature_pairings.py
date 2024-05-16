import numpy as np

path_save_file = "/cluster/home/emmalei/Master_project_network/feature_vectors/validation/"

pairing_array = []
pairing_array.append(["2d","3d","label"]) # labels for the .csv file

expressions = ["ANG", "DIS", "FEA", "HAP", "SAD", "SUR"]
F_rulette = 2
M_rulette = 1

# Females
for i in range(1, 59):
    for j in range(6):
        for k in range(1, 5):
            # Pair with self:
            pairing_array.append([f"2d/feat2d_F{i:02d}{expressions[j]}_{k}.pt", f"3d/feat3d_F{i:02d}{expressions[j]}_{k}.pt", str(1)])
            
            if k == 1:
                # Match with similar expression and more intense expression
                pairing_array.append([f"2d/feat2d_F{i:02d}{expressions[j]}_{1}.pt", f"3d/feat3d_F{i:02d}{expressions[j]}_{2}.pt", str(1)])
                pairing_array.append([f"2d/feat2d_F{i:02d}{expressions[j]}_{1}.pt", f"3d/feat3d_F{i:02d}{expressions[j]}_{3}.pt", str(1)])
            
            if k == 4:
                # Match with similar expression and less intense expression
                pairing_array.append([f"2d/feat2d_F{i:02d}{expressions[j]}_{4}.pt", f"3d/feat3d_F{i:02d}{expressions[j]}_{2}.pt", str(1)])
                pairing_array.append([f"2d/feat2d_F{i:02d}{expressions[j]}_{4}.pt", f"3d/feat3d_F{i:02d}{expressions[j]}_{3}.pt", str(1)])
                
                # Vertical steps across expressions within same ID (extreme intensity edition)
                if j != 5:
                    for l in range(j+1, 6):
                        pairing_array.append([f"2d/feat2d_F{i:02d}{expressions[j]}_{4}.pt", f"3d/feat3d_F{i:02d}{expressions[l]}_{4}.pt", str(1)])
                        
            # Pair with other identity:
            # Other female, same expression and expression intensity
            # Skip if same identity as current
            if F_rulette == i:
                F_rulette += 1
            
            # Reset when surpassed number of registered female IDs 
            if F_rulette == 59:
                F_rulette = 1
               
            pairing_array.append([f"2d/feat2d_F{i:02d}{expressions[j]}_{k}.pt", f"3d/feat3d_F{F_rulette:02d}{expressions[j]}_{k}.pt", str(-1)])
            
            F_rulette += 1
            
            # Other male, same expression and expression intensity
            # Reset when surpassed number of registered male IDs 
            if M_rulette == 44:
                M_rulette = 1
               
            pairing_array.append([f"2d/feat2d_F{i:02d}{expressions[j]}_{k}.pt", f"3d/feat3d_M{M_rulette:02d}{expressions[j]}_{k}.pt", str(-1)])
            
            M_rulette += 1
            
    print(f"Finished F{i:02d}")


# Males
for i in range(1, 44):
    for j in range(6):
        for k in range(1, 5):
            # Pair with self
            pairing_array.append([f"2d/feat2d_M{i:02d}{expressions[j]}_{k}.pt", f"3d/feat3d_M{i:02d}{expressions[j]}_{k}.pt", str(1)])
             
            if k == 1:
                # Match with similar expression and more intense expression
                pairing_array.append([f"2d/feat2d_M{i:02d}{expressions[j]}_{1}.pt", f"3d/feat3d_M{i:02d}{expressions[j]}_{2}.pt", str(1)])
                pairing_array.append([f"2d/feat2d_M{i:02d}{expressions[j]}_{1}.pt", f"3d/feat3d_M{i:02d}{expressions[j]}_{3}.pt", str(1)])
            
            if k == 4:
                # Match with similar expression and less intense expression
                pairing_array.append([f"2d/feat2d_M{i:02d}{expressions[j]}_{4}.pt", f"3d/feat3d_M{i:02d}{expressions[j]}_{2}.pt", str(1)])
                pairing_array.append([f"2d/feat2d_M{i:02d}{expressions[j]}_{4}.pt", f"3d/feat3d_M{i:02d}{expressions[j]}_{3}.pt", str(1)])
                
                # Vertical steps across expressions within same ID (extreme intensity edition)
                if j != 5:
                    for l in range(j+1, 6):
                        pairing_array.append([f"2d/feat2d_M{i:02d}{expressions[j]}_{4}.pt", f"3d/feat3d_M{i:02d}{expressions[l]}_{4}.pt", str(1)])
               
            # Pair with other identity:
            # Other female, same expression and expression intensity
            # Reset when surpassed number of registered female IDs 
            if F_rulette == 59:
                F_rulette = 1
               
            pairing_array.append([f"2d/feat2d_M{i:02d}{expressions[j]}_{k}.pt", f"3d/feat3d_F{F_rulette:02d}{expressions[j]}_{k}.pt", str(-1)])
            
            F_rulette += 1
            
            # Other male, same expression and expression intensity
            # Skip if same identity as current
            if M_rulette == i:
                M_rulette += 1
            
            # Reset when surpassed number of registered male IDs 
            if M_rulette == 44:
                M_rulette = 1
               
            pairing_array.append([f"2d/feat2d_M{i:02d}{expressions[j]}_{k}.pt", f"3d/feat3d_M{M_rulette:02d}{expressions[j]}_{k}.pt", str(-1)])
            
            M_rulette += 1
            
    print(f"Finished M{i:02d}")

print(f"Number of pairings {len(pairing_array)}")
#print(pairing_array)

# Save the array as a .csv
np.savetxt(f"{path_save_file}feature_pairings.csv", pairing_array, delimiter=",", fmt='%s')
print(f"Feature pairings saved to {path_save_file}feature_pairings.csv")