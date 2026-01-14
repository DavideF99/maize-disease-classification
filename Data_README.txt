The image data is contained under static/leaf_images/*
These are the high resolution, original images. Feel free to resize them as you please.

Accompanying these image files is a CSV titled "Database.csv" which contains the class labels for the dataset. 
It follows the following format:
image_id, image_file_path, image_class_label.

A single image can contain multiple labels and this is reflected in the csv through multiple rows for the same image. For example:
1220,/static/leaf_images/2009-03-24 Maize GLS images (Baynesfield KZN)/110_CIMG3846_1-109-1.JPG,1
1220,/static/leaf_images/2009-03-24 Maize GLS images (Baynesfield KZN)/110_CIMG3846_1-109-1.JPG,2

Classes are identified using numerical values and these correspond to diseases/other features in the following way:
1,'Grey Leaf Spot'
2,'Northern Corn Leaf Blight'
3,'Phaeosphaeria Leaf Spot'
4,'Common Rust'
5,'Southern Rust'
6,'Healthy'
7,'Other'
8,'Unknown'

Classes 7 and 8 may require additional clarification. 'Other' refers to an additional feature that is not listed here. This may refer to excessive leaf tearing, presence of insects and insect damage. 'Unknown' refers to the case where a maize leaf is believed to have a disease that does not belong to those listed.
