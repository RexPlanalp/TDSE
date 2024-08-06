import json
import numpy as np
from collections import deque

from petsc4py import PETSc
comm = PETSc.COMM_WORLD
rank = comm.rank

import time

class Sim:
    def __init__(self,file_name):
        data = self.openJSON(file_name)
        for key, value in data.items():
            setattr(self, key, value)

    def __str__(self):
        return '\n'.join(f"{key}: {getattr(self, key)}" for key in self.__dict__)

    def openJSON(self,file_name):
        with open(file_name, 'r') as file:
            input_par = json.load(file)
        return input_par
    
    def find_reachable_points(self, delta_l, delta_m):
        def is_valid(l, m):
            return 0 <= l <= self.lm["lmax"] and -l <= m <= l

        queue = deque([(self.state[1], self.state[2])])
        reachable_points = set([(self.state[1], self.state[2])])
        index_to_point = {}
        point_to_index = {}
        index = 0

        # Add initial state to dictionaries
        initial_l, initial_m = self.state[1], self.state[2]
        index_to_point[index] = (initial_l, initial_m)
        point_to_index[(initial_l, initial_m)] = index
        index += 1

        while queue:
            current_l, current_m = queue.popleft()
            for dl in delta_l:
                for dm in delta_m:
                    new_l = current_l + dl
                    new_m = current_m + dm
                    if is_valid(new_l, new_m) and (new_l, new_m) not in reachable_points:
                        reachable_points.add((new_l, new_m))
                        queue.append((new_l, new_m))
                        
                        # Add to dictionaries
                        index_to_point[index] = (new_l, new_m)
                        point_to_index[(new_l, new_m)] = index
                        index += 1
        
        return point_to_index, index_to_point

    # def lm_block_maps(self):
    #     if self.laser["polarization"] == "linear":
    #         lmax = self.lm["lmax"]
    #         m_value = self.state[2]

    #         lm_dict = {}
    #         for l in range(lmax+1):
    #             lm_dict[(l,m_value)] = l
    #         block_dict = {value: key for key, value in lm_dict.items()}
    #         self.lm_dict,self.block_dict = lm_dict,block_dict
    #     elif self.laser["polarization"]  == "elliptical":
    #         lmax = self.lm["lmax"]
    #         def block_number(l, m):
    #             sum_blocks = sum(2*i + 1 for i in range(l))
    #             m_offset = m + l
    #             return sum_blocks + m_offset
    #         lm_dict = {}
    #         for l in range(lmax+1):
    #             for m in range(-l,l+1):
    #                 lm_dict[(l,m)] = block_number(l,m)
    #         block_dict = {value: key for key, value in lm_dict.items()}
    #         self.lm_dict,self.block_dict = lm_dict,block_dict

    def lm_block_maps(self):
        if self.laser["polarization"] == "linear" and self.laser["ell"] == 0:
            delta_l = [1,-1]
            delta_m = [0]
            self.lm_dict, self.block_dict = self.find_reachable_points(delta_l, delta_m)

        elif self.laser["polarization"]  == "circular" and (self.laser["ell"] == 1 or self.laser["ell"] == -1):
            if self.laser["ell"] == 1:
                delta_l = [1,-1]
                delta_m = [1]
            elif self.laser["ell"] == -1:
                delta_l = [1,-1]
                delta_m = [-1]
            self.lm_dict, self.block_dict = self.find_reachable_points(delta_l, delta_m)

        elif self.laser["polarization"] == "elliptical" and not (self.laser["ell"] == 1 or self.laser["ell"] == -1):
            delta_l = [1,-1]
            delta_m = [1,-1,0]
            self.lm_dict, self.block_dict = self.find_reachable_points(delta_l, delta_m)


    def calc_n_block(self):
        self.n_block = len(self.lm_dict)

    def timeGrid(self):
        self.tau = 2*np.pi/self.laser["w"]
        self.time_size = self.box["N"]*self.tau
        self.Nt = int(np.rint(self.time_size/ self.box["time_spacing"])) + 1

        self.post_time_size = self.box["N_post"] * self.tau
        self.Nt_post = int(np.rint(self.post_time_size/ self.box["time_spacing"])) + 1 
    def spacialGrid(self):
        self.Nr = int(self.box["grid_size"]/self.box["grid_spacing"])+1
    def printGrid(self):
        if rank == 0:
            print(f"Spacial Grid: [{0},{self.box['grid_size']}]")
            print("\n")
            print(f"Temporal Grid: [{0},{self.time_size}, dt = {self.box['time_spacing']}]")
            print("\n")
            print(f"Post Propagation: [{self.time_size},{self.time_size+self.post_time_size}, dt = {self.box['time_spacing']}]")
            print("\n") 




    def gather_matrix(self,M, comm, root):
        local_csr = M.getValuesCSR()
        local_indptr, local_indices, local_data = local_csr

        gathered_indices = comm.gather(local_indices, root=root)
        gathered_data = comm.gather(local_data, root=root)
        gathered_indptr = comm.gather(local_indptr, root=root)

        if comm.rank == root:
            global_indices = np.concatenate(gathered_indices).astype(np.int32)
            global_data = np.concatenate(gathered_data)
            global_indptr = [gathered_indptr[0]]
            offset = global_indptr[0][-1]
            for indptr in gathered_indptr[1:]:
                corrected_indptr = indptr[1:] + offset 
                global_indptr.append(corrected_indptr)
                offset += indptr[-1] - indptr[0]
            global_indptr = np.concatenate(global_indptr)
            return PETSc.Mat().createAIJWithArrays([M.getSize()[0], M.getSize()[1]], (global_indptr, global_indices, global_data), comm=PETSc.COMM_SELF)
        return None

    def kron(self,A, B, comm,nonzeros):
        comm = comm.tompi4py()
        root = 0
        # Gather the matrices to the root
        rootA = self.gather_matrix(A, comm, root)
        rootB = self.gather_matrix(B, comm, root)

        if comm.rank == root:
            # Compute the Kronecker product only on the root
            rootC = rootA.kron(rootB)
            rootA.destroy()
            rootB.destroy()
            viewer = PETSc.Viewer().createBinary("temp/temp.bin","w",comm = PETSc.COMM_SELF)
            rootC.view(viewer)
            viewer.destroy()
        else:
            rootC = None
        
        C = PETSc.Mat(comm = PETSc.COMM_WORLD,nnz = nonzeros)
        viewer = PETSc.Viewer().createBinary('temp/temp.bin', 'r',comm = PETSc.COMM_WORLD)
        C.load(viewer)
        viewer.destroy()
        return C


if __name__ == "__main__":
    start = time.time()
    simInstance = Sim("input_new.json")  
    simInstance.lm_block_maps() 
    simInstance.calc_n_block()   
    simInstance.timeGrid() 
    simInstance.spacialGrid() 
    end = time.time()
    print(end-start)