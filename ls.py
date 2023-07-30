"""
The code presented here is mostly extracted from the GitHub repository: https://github.com/

The full publication is cited:


"""
from growingspheres import GrowingSpheres
import numpy as np
import math
import time

def ls_method(perturbator, data, ioi, ioi_label, regul = 0.1, seed_sel = None, robustness_call = False):
    """
    Method that creates random perturbations based on the nearest neighbors (Laugel, Renard et al. Defining Locality for Surrogates in Post-hoc Interpretability, 2018)
    """
    def normal_nn(center, instance_array, instance_targets = None):
        """
        Method that returns the array of instances and distances of an array of instances w.r.t. a center instance
        """
        distance = []
        for i in range(instance_array.shape[0]):
            dist = np.sqrt(np.sum((instance_array[i,:]-center)**2))
            if instance_targets is None:
                distance.append((instance_array[i,:],dist))
            else:
                distance.append((instance_array[i,:],instance_targets[i],dist))
        distance.sort(key=lambda x: x[-1])
        return distance

    def get_nearest_train_neighbors_to_cf(distance_cf_to_x_border,r):
        """
        Method that extracts the nearest neighbor training instances to the closest cf instance in the border found.
        """
        if distance_cf_to_x_border[0][-1] > r:
            r_10_percent = int(len(distance_cf_to_x_border)*0.1)
            r = distance_cf_to_x_border[r_10_percent][-1]
        train_with_target_in_r_ball = [(i[0],i[1]) for i in distance_cf_to_x_border if i[-1] <= r]
        if len(np.unique([i[1] for i in train_with_target_in_r_ball])) == 1:
            unique_label = train_with_target_in_r_ball[0][1]
            count = 0
            for i in distance_cf_to_x_border:
                if i[1] != unique_label:
                    train_with_target_in_r_ball.append((i[0],i[1]))
                    count += 1
                    if count == 5:
                        break
        train_instances = np.array([i[0] for i in train_with_target_in_r_ball])
        return train_instances

    def generate_sphere(sphere_center,r,cf_gen=False):
        """
        Method that generates instances on the surface of a sphere of radius r
        """
        n = len(perturbator.features)
        if cf_gen:
            points_number = int(np.round(100*r*n*(math.pi**(n/4)))) # 500
        else:
            points_number = int(np.round(100*n*(math.pi**(n/4))))
        # First assume all features continuous
        random_points = np.random.uniform(0,1,size=(points_number,len(perturbator.processed_features)))
        random_vectors = random_points - sphere_center
        random_vectors_magnitude = np.linalg.norm(random_vectors,axis=1)
        norm_vectors = random_vectors / random_vectors_magnitude.reshape(-1,1)
        sphere_points = sphere_center + norm_vectors*r
        return sphere_points
        
    def find_sphere_cf(sphere_points,label):
        """
        Method that outputs the perturbations on the sphere surface that changed label w.r.t. IOI.
        """
        sphere_found_cf = False
        sphere_all = np.copy(sphere_points)
        sphere_cf = np.copy(sphere_points)
        if sphere_all.shape[0] > 0:
            sphere_cf = sphere_points[np.where(perturbator.global_model.call_predict(sphere_points) != label)]
            if sphere_cf.shape[0] > 0:
                sphere_found_cf = True
        return sphere_cf, sphere_found_cf, sphere_all

    def sphere_construction(center,r,label,cf_gen=False):
        """
        Method that generates, adjusts, and tries to find cf points on a sphere surface of radius r
        """
        sphere = generate_sphere(center,r,cf_gen)
        sphere_cf, sphere_cf_bool, sphere_all = find_sphere_cf(sphere,label)
        return sphere_cf, sphere_cf_bool, sphere_all
    
    def optimal_sphere_construction(sphere_cf, sphere_all, center,r_0, r_1, label):
        """
        Recursive binary search method to find the closest sphere with cf points on its surface
        """
        if r_1 - r_0 <= 0.01:
            return sphere_cf, sphere_all
        else:
            sphere_0_cf, sphere_0_bool, sphere_0_all = sphere_construction(center,r_0,label)
            sphere_1_cf, sphere_1_bool, sphere_1_all = sphere_construction(center,r_1,label)
            if not sphere_0_bool and not sphere_1_bool:
                # print(f'Sphere instances do not have both classes. r_0: {r_0}, r_1: {r_1}')
                return sphere_cf, sphere_all
            elif sphere_0_bool:
                sphere_cf, sphere_all = sphere_0_cf, sphere_0_all
                return sphere_cf, sphere_all
            elif not sphere_0_bool and sphere_1_bool:
                r_mid = np.mean([r_0,r_1]) 
                sphere_mid_cf, sphere_mid_bool, sphere_mid_all = sphere_construction(center,r_mid,label)
                if sphere_mid_bool:
                    r_1 = r_mid
                    sphere_cf, sphere_all = sphere_mid_cf, sphere_mid_all
                else:
                    r_0 = r_mid
                    sphere_cf, sphere_all = sphere_1_cf, sphere_1_all
                sphere_cf, sphere_all = optimal_sphere_construction(sphere_cf,sphere_all,center,r_0,r_1,label)
            elif not sphere_1_bool:
                print(f'Outer radius without CF instances: r_1 = {r_1}, r_0 = {r_0}, diff. = {r_1 - r_0}')
                return sphere_cf, sphere_all
        return sphere_cf, sphere_all 

    def initialize_radius():
        """
        Method to initialize the center, r_0, r_1, values used for the sphere construction
        """
        center, r_0 = ioi.normal_x.values.reshape(1, -1), 0.01
        r_1 = 1
        sphere_r1_cf_exist = False
        while not sphere_r1_cf_exist:
            sphere_1_cf, sphere_r1_cf_exist, sphere_1_all = sphere_construction(center,r_1, ioi_label)
            r_1 = 0.99*r_1
            if np.isclose(r_1,r_0,0.01) and not sphere_r1_cf_exist:
                print(f'Could not find a CF. Generating neighborhood around ioi')
                break
        if not sphere_r1_cf_exist:
            sphere_cf_array, sphere_all_array = optimal_sphere_construction(center, center, center, r_0, r_1, ioi_label)
        else:
            sphere_cf_array, sphere_all_array = optimal_sphere_construction(sphere_1_cf, sphere_1_all, center, r_0, r_1, ioi_label)
        if sphere_cf_array.shape[0] == 1:
            if np.equal(sphere_cf_array,center).all():
                print(f'Optimal sphere could not construct feasible sphere!: center == sphere_cf_array')
        return sphere_cf_array, sphere_all_array

    if seed_sel is None:
        np.random.seed(perturbator.seed_int)
    else:
        np.random.seed(seed_sel)

    start_time = time.time()
    # sphere_cf_array, sphere_all_array = initialize_radius()
    # initialize_radius_time = time.time()
    # distance = normal_nn(ioi.normal_x,sphere_cf_array)
    # nearest_sphere_cf_to_x = distance[0][0]
    growing_sphere = GrowingSpheres(ioi.normal_x.values.reshape(1,-1), perturbator.global_model.call_predict, target_class = 1 - ioi_label, n_in_layer = 2000, first_radius = 0.1, decrease_radius = 10, verbose = True)
    nearest_sphere_cf_to_x = growing_sphere.find_counterfactual()
    sort_cf_instances_time = time.time()
    r_gen = 0.3 # Original is simply 0.3 (no change with respect to features in the dataset)
    distance_near_cf_to_x_border = normal_nn(nearest_sphere_cf_to_x, data.processed_train_pd.values, data.train_target.values)
    sort_train_instances_time = time.time()
    near_cf_to_x_border_train = get_nearest_train_neighbors_to_cf(distance_near_cf_to_x_border,r_gen)
    generate_neighbors_time = time.time()
    weights = np.ones((near_cf_to_x_border_train.shape[0],))
    weighting_time = time.time()
    total_time = weighting_time - start_time

    # print(f'LS   Step 1: Sort CF instances             : {np.round(sort_cf_instances_time - start_time, 4)} (s) / ({np.round((sort_cf_instances_time - start_time)*100/total_time, 2)}%)')
    # print(f'LS   Step 2: Sort train instances          : {np.round(sort_train_instances_time - sort_cf_instances_time, 4)} (s) / ({np.round((sort_train_instances_time - sort_cf_instances_time)*100/total_time, 2)}%)')
    # print(f'LS   Step 3: Neighbors generation          : {np.round(generate_neighbors_time - sort_train_instances_time, 4)} (s) / ({np.round((generate_neighbors_time - sort_train_instances_time)*100/total_time, 2)}%)')
    # print(f'LS   Step 4: Weighting                     : {np.round(weighting_time - generate_neighbors_time, 4)} (s) / ({np.round((weighting_time - generate_neighbors_time)*100/total_time, 2)}%)')
    # print(f'LS   Total                                 : {np.round(total_time, 4)} (s)')

    return near_cf_to_x_border_train, near_cf_to_x_border_train, weights, total_time