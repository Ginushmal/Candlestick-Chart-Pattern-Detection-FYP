# create a pytorch dataset to load the label_df and ohcl_segments_df

import torch
import os
import pandas as pd

class OHCLDataset(torch.utils.data.Dataset):
    def __init__(self, label_df, ohcl_segments_df,S=7, B=2,C=8, transform=None):
        self.label_df = label_df
        self.ohcl_segments_df = ohcl_segments_df
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        # get the number of level 0 indexes in the label_df
        num_instances = self.label_df.index.get_level_values('Instance').nunique()
        
        return num_instances

    def __getitem__(self, index):
        print("Index : ", index)
        print("label_df : ", self.label_df)
        
        # Extract rows using .loc
        label_row = self.label_df.loc[index].copy()
        ohcl_row = self.ohcl_segments_df.loc[index].copy()
        
        print("label_row : ", label_row)
        print("ohcl_row : ", ohcl_row)

        # Convert the Pattern column to int in the label_row
        label_row['Pattern'] = label_row['Pattern'].astype(int)

        # Convert to tensors
        # label = torch.tensor(label_row.values)
        ohcl = torch.tensor(ohcl_row.values)
        
        
        label_matrix = torch.zeros((self.S, self.C + self.B * 3))

        # iterate through each row of the labele pandas dataframe
        for index, row in label_row.iterrows():
            pattern = row["Pattern"]
            center = row["Center"]
            width = row["Width"]
            
            old_center = center
            
            # get the cell index
            cell_idx = int(center * self.S) 
            # bbox center relative to the cell
            center = (center * self.S) - cell_idx
            
            # bbox width relative to the cell
            width = width * self.S
            
            # print("--------------------------------------------------------------")
            # print("label_matrix before crash======++++ : ", label_matrix)
            
            try:
                if label_matrix[cell_idx, 8] == 0:
                    # one-hot encoding for class label
                    label_matrix[cell_idx, int(pattern)] = 1
                    # objectness score
                    label_matrix[cell_idx, 8] = 1
                    # center of the bbox
                    label_matrix[cell_idx, 9] = center
                    # width of the bbox
                    label_matrix[cell_idx, 10] = width
            except Exception as e:
                print("Errornous Data : " ,label_matrix)
                print('old_center : ', old_center , 'center : ', center, 'width : ', width , 'pattern : ', pattern , 'cell_idx : ', cell_idx)
                print("Index : ", index)
                print("shhhhhhhhhiiiiiiiiiiiiitttttttttttttttt")
                raise e
            
                
            # print("--------------------------------------------------------------")
            # print("cell_idx : ", cell_idx)
            # print("center : ", center)
            # print("width : ", width)
            # print("pattern : ", pattern)
            # print("label_matrix : ", label_matrix)
        
        # Reshape OHCL data to (4, -1)
        if ohcl.shape[0] % 4 != 0:
            raise ValueError(f"Inconsistent OHCL data shape for index {index}. Cannot reshape to (4, -1).")
        ohcl_reshaped = ohcl.T.reshape(4, -1)


        if self.transform:
            label_matrix, ohcl = self.transform(label_matrix, ohcl_reshaped)
        
        ohcl_reshaped = ohcl_reshaped.float()
        label_matrix = label_matrix.float()  # Convert input to float32
        
        # print("label_matrix in dataset.py", label_matrix)
        # print("ohcl_reshaped in dataset.py", ohcl_reshaped)


            
        return ohcl_reshaped , label_matrix