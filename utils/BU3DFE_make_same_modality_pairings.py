import numpy as np

path_save_file = "/cluster/home/emmalei/Master_project_network/feature_vectors/test/"

pairing_array = []
pairing_array.append(["frame1","frame2","dimention","label"]) # labels for the .csv file

expressions = ["ANG", "DIS", "FEA", "HAP", "SAD", "SUR"] #NEU
F_rulette = 2
M_rulette = 1

# Females
for i in range(1, 57):
    for j in range(6):
        for k in range(1, 5):
            # Pair with self:
            for l in range(1,5):
                if l != k:
                    pairing_array.append([f"2d/feat2d_F{i:02d}{expressions[j]}_{k}.pt", f"2d/feat2d_F{i:02d}{expressions[j]}_{l}.pt", str(2), str(1)])
                    pairing_array.append([f"3d/feat3d_F{i:02d}{expressions[j]}_{k}.pt", f"3d/feat3d_F{i:02d}{expressions[j]}_{l}.pt", str(3), str(1)])
                    
            
            # Pair with neutral
            pairing_array.append([f"2d/feat2d_F{i:02d}{expressions[j]}_{k}.pt", f"2d/feat2d_F{i:02d}NEU_0.pt", str(2), str(1)])
            pairing_array.append([f"3d/feat3d_F{i:02d}{expressions[j]}_{k}.pt", f"3d/feat3d_F{i:02d}NEU_0.pt", str(3), str(1)])
            
            for exp in expressions:
                if exp != expressions[j]:
                    pairing_array.append([f"2d/feat2d_F{i:02d}{expressions[j]}_{k}.pt", f"2d/feat2d_F{i:02d}{exp}_{k}.pt", str(2), str(1)])
                    pairing_array.append([f"3d/feat3d_F{i:02d}{expressions[j]}_{k}.pt", f"3d/feat3d_F{i:02d}{exp}_{k}.pt", str(3), str(1)])
                    
                    
            # Pair with other identity:
            # Other female, same expression and expression intensity
            for l in range(4):
                # Skip if same identity as current
                if F_rulette == i:
                    F_rulette += 1
                
                # Reset when surpassed number of registered female IDs 
                if F_rulette == 57:
                    F_rulette = 1
                
                pairing_array.append([f"2d/feat2d_F{i:02d}{expressions[j]}_{k}.pt", f"2d/feat2d_F{F_rulette:02d}{expressions[j]}_{k}.pt", str(2), str(0)])
                pairing_array.append([f"3d/feat3d_F{i:02d}{expressions[j]}_{k}.pt", f"3d/feat3d_F{F_rulette:02d}{expressions[j]}_{k}.pt", str(3), str(0)])
                
                F_rulette += 1
                
                # Other male, same expression and expression intensity
                # Reset when surpassed number of registered male IDs 
                if M_rulette == 45:
                    M_rulette = 1
                
                pairing_array.append([f"2d/feat2d_F{i:02d}{expressions[j]}_{k}.pt", f"2d/feat2d_M{M_rulette:02d}{expressions[j]}_{k}.pt", str(2), str(0)])
                pairing_array.append([f"3d/feat3d_F{i:02d}{expressions[j]}_{k}.pt", f"3d/feat3d_M{M_rulette:02d}{expressions[j]}_{k}.pt", str(3), str(0)])
                
                M_rulette += 1
                
    # Pair neutral
    # self
    #pairing_array.append([f"2d/feat2d_F{i:02d}NEU_0.pt", f"3d/feat3d_F{i:02d}NEU_0.pt", str(1)])
    # Other identities
    for l in range(4):
        # Skip if same identity as current
        if F_rulette == i:
            F_rulette += 1
        
        # Reset when surpassed number of registered female IDs 
        if F_rulette == 57:
            F_rulette = 1
        
        pairing_array.append([f"2d/feat2d_F{i:02d}NEU_0.pt", f"2d/feat2d_F{F_rulette:02d}NEU_0.pt", str(2), str(0)])
        pairing_array.append([f"3d/feat3d_F{i:02d}NEU_0.pt", f"3d/feat3d_F{F_rulette:02d}NEU_0.pt", str(3), str(0)])
        
        F_rulette += 1
        
        # Other male, same expression and expression intensity
        # Reset when surpassed number of registered male IDs 
        if M_rulette == 45:
            M_rulette = 1
        
        pairing_array.append([f"2d/feat2d_F{i:02d}NEU_0.pt", f"2d/feat2d_M{M_rulette:02d}NEU_0.pt", str(2), str(0)])
        pairing_array.append([f"3d/feat3d_F{i:02d}NEU_0.pt", f"3d/feat3d_M{M_rulette:02d}NEU_0.pt", str(3), str(0)])
        
        M_rulette += 1
    
    print(f"Finished F{i:02d}")


# Males
for i in range(1, 45):
    for j in range(6):
        for k in range(1, 5):
            # Pair with self and rest of horisontal
            for l in range(1,5):
                if l != k:
                    pairing_array.append([f"2d/feat2d_M{i:02d}{expressions[j]}_{k}.pt", f"2d/feat2d_M{i:02d}{expressions[j]}_{l}.pt", str(2), str(1)])
                    pairing_array.append([f"3d/feat3d_M{i:02d}{expressions[j]}_{k}.pt", f"3d/feat3d_M{i:02d}{expressions[j]}_{l}.pt", str(3), str(1)])

            # Pair with neutral
            pairing_array.append([f"2d/feat2d_M{i:02d}{expressions[j]}_{k}.pt", f"2d/feat2d_M{i:02d}NEU_0.pt", str(2), str(1)])
            pairing_array.append([f"3d/feat3d_M{i:02d}{expressions[j]}_{k}.pt", f"3d/feat3d_M{i:02d}NEU_0.pt", str(3), str(1)])
            
            
            for exp in expressions:
                if exp != expressions[j]:
                    pairing_array.append([f"2d/feat2d_M{i:02d}{expressions[j]}_{k}.pt", f"2d/feat2d_M{i:02d}{exp}_{k}.pt", str(2), str(1)])
                    pairing_array.append([f"3d/feat3d_M{i:02d}{expressions[j]}_{k}.pt", f"3d/feat3d_M{i:02d}{exp}_{k}.pt", str(3), str(1)])
            
            # Pair with other identity:
            # Other female, same expression and expression intensity
            for l in range(4):
                # Reset when surpassed number of registered female IDs 
                if F_rulette == 57:
                    F_rulette = 1
                
                pairing_array.append([f"2d/feat2d_M{i:02d}{expressions[j]}_{k}.pt", f"2d/feat2d_F{F_rulette:02d}{expressions[j]}_{k}.pt", str(2), str(0)])
                pairing_array.append([f"3d/feat3d_M{i:02d}{expressions[j]}_{k}.pt", f"3d/feat3d_F{F_rulette:02d}{expressions[j]}_{k}.pt", str(3), str(0)])
                
                F_rulette += 1
                
                # Other male, same expression and expression intensity
                # Skip if same identity as current
                if M_rulette == i:
                    M_rulette += 1
                
                # Reset when surpassed number of registered male IDs 
                if M_rulette == 45:
                    M_rulette = 1
                
                pairing_array.append([f"2d/feat2d_M{i:02d}{expressions[j]}_{k}.pt", f"2d/feat2d_M{M_rulette:02d}{expressions[j]}_{k}.pt", str(2), str(0)])
                pairing_array.append([f"3d/feat3d_M{i:02d}{expressions[j]}_{k}.pt", f"3d/feat3d_M{M_rulette:02d}{expressions[j]}_{k}.pt", str(3), str(0)])
                
                M_rulette += 1

    # Pair neutral
    # self
    #pairing_array.append([f"2d/feat2d_M{i:02d}NEU_0.pt", f"3d/feat3d_M{i:02d}NEU_0.pt", str(1)])
    # Other identities
    # Other female, same expression and expression intensity
    for l in range(4):
        # Reset when surpassed number of registered female IDs 
        if F_rulette == 57:
            F_rulette = 1
        
        pairing_array.append([f"2d/feat2d_M{i:02d}NEU_0.pt", f"2d/feat2d_F{F_rulette:02d}NEU_0.pt", str(2), str(0)])
        pairing_array.append([f"3d/feat3d_M{i:02d}NEU_0.pt", f"3d/feat3d_F{F_rulette:02d}NEU_0.pt", str(3), str(0)])
        
        F_rulette += 1
        
        # Other male, same expression and expression intensity
        # Skip if same identity as current
        if M_rulette == i:
            M_rulette += 1
        
        # Reset when surpassed number of registered male IDs 
        if M_rulette == 45:
            M_rulette = 1
        
        pairing_array.append([f"2d/feat2d_M{i:02d}NEU_0.pt", f"2d/feat2d_M{M_rulette:02d}NEU_0.pt", str(2), str(0)])
        pairing_array.append([f"3d/feat3d_M{i:02d}NEU_0.pt", f"3d/feat3d_M{M_rulette:02d}NEU_0.pt", str(3), str(0)])
        
        M_rulette += 1
    print(f"Finished M{i:02d}")
    
print(f"Number of pairings {len(pairing_array)}")
#print(pairing_array)

# Save the array as a .csv
np.savetxt(f"{path_save_file}same_modality_feature_pairings.csv", pairing_array, delimiter=",", fmt='%s')
print(f"Feature pairings saved to {path_save_file}same_modality_feature_pairings.csv")