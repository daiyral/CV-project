import pickle
import numpy as np
import torch



# Function to load the image from a pickle file
def load_image_from_pickle(file_path):
    try:
        # Attempt to load using torch.load with weights_only=True
        print("Trying to load with torch.load...")
        image = torch.load(file_path, weights_only=True)
        print("Loaded with torch!")
    except Exception as e_torch:
        print(f"torch.load failed: {e_torch}")

        try:
            # Fallback to loading with pickle if torch.load fails
            print("Trying to load with pickle.load...")
            with open(file_path, 'rb') as f:
                image = pickle.load(f)
            print("Loaded with pickle!")
        except Exception as e_pickle:
            print(f"pickle.load failed: {e_pickle}")
            return None  # Both methods failed, return None

    # Check if the loaded object is a tensor or ndarray of tensors
    if isinstance(image, torch.Tensor):
        print("Loaded object is a PyTorch tensor.")
        # Convert tensor to a NumPy array
        image = image.cpu().numpy()  # Ensure tensor is on CPU and convert to numpy array
    elif isinstance(image, np.ndarray):
        print("Loaded object is a NumPy ndarray.")
        # If it's a ndarray of tensors, convert each tensor to a NumPy array
        if isinstance(image[0], torch.Tensor):  # Check if the first element is a tensor
            print("Image is an ndarray of tensors.")
            image = np.array([tensor.cpu().numpy() for tensor in image])
        elif isinstance(image[0], np.ndarray):
            # If it's already in NumPy format
            print("Image is an ndarray of NumPy arrays.")
            pass
        else:
            print(f"Unsupported ndarray content type: {type(image[0])}")
            return None
    elif isinstance(image, list):
        print("Loaded object is a list.")
        # Convert list to a NumPy array if possible
        image = np.array([np.array(item) if isinstance(item, torch.Tensor) else item for item in image])
    else:
        print(f"Loaded object is of type {type(image)} which is not supported.")
        return None

    return image


def filter_by_opacity(image, opacity_threshold=0.1):
    # List to store all xyz coordinates
    xyz_list = []

    # Loop through the array and extract 'xyz' data
    # Loop through the array and extract 'xyz' data
    for entry in image:
        if 'xyz' in entry and 'opacity' in entry:
            xyz_tensor = entry['xyz']  # Extract the tensor
            opacity_tensor = entry['opacity']  # Extract the opacity tensor

            # Convert both tensors to NumPy arrays in one go
            xyz_array = xyz_tensor.cpu().numpy()
            opacity_array = opacity_tensor.cpu().numpy().flatten()  # Flatten the opacity tensor

            # Create a mask where opacity > 0.001
            mask = opacity_array > opacity_threshold

            # Filter xyz_array using the mask and append the filtered points
            xyz_filtered = xyz_array[:, mask]  # Apply mask to select points with valid opacity
            xyz_list.extend(xyz_filtered[0])

    return xyz_list


def visualize_3d_points(x_sample, y_sample, z_sample):
    import plotly.graph_objects as go
    # Create a 3D scatter plot with the sampled points
    fig = go.Figure(data=[go.Scatter3d(
        x=x_sample,
        y=y_sample,
        z=z_sample,
        mode='markers',
        marker=dict(size=2, color='blue')
    )])

    # Set axis labels and title
    fig.update_layout(
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ),
        title='3D Point Cloud Visualization (Sampled 10,000 points)'
    )

    # Show the plot
    fig.write_html("3d_point_cloud_sampled_10000.html")
    fig.show()


def sample_points(xyz_combined_flattened, sample_size=10000):
    # Randomly sample 10,000 points from the combined data
    if xyz_combined_flattened.shape[0] > sample_size:
        sampled_indices = np.random.choice(xyz_combined_flattened.shape[0], sample_size, replace=False)
        x_sample = xyz_combined_flattened[sampled_indices, 0]
        y_sample = xyz_combined_flattened[sampled_indices, 1]
        z_sample = xyz_combined_flattened[sampled_indices, 2]
    else:
        # If there are fewer than 10,000 points, use all points
        x_sample = xyz_combined_flattened[:, 0]
        y_sample = xyz_combined_flattened[:, 1]
        z_sample = xyz_combined_flattened[:, 2]

    return x_sample, y_sample, z_sample


def get_filtered_points_with_opacity(image, opacity_threshold=0.2, sample_size=10000):
    xyz_list = filter_by_opacity(image, opacity_threshold=opacity_threshold)

    # Combine all xyz data into a single NumPy array
    xyz_combined = np.vstack(xyz_list)  # Stack all xyz arrays vertically

    # Now xyz_combined contains all the 3D points for visualization
    print(xyz_combined)

    xyz_combined_flattened = xyz_combined.reshape(-1, 3)  # (11534336, 3)

    # If xyz_combined is not empty, proceed to visualization
    if xyz_combined.size > 0:

        # Sample a smaller number of points for visualization
        x_sample, y_sample, z_sample = sample_points(xyz_combined_flattened, sample_size=sample_size)

        return x_sample, y_sample, z_sample
    else:
        return None, None, None

# Function to create the Dash app
def create_dash_app(x_sample, y_sample, z_sample):
    import plotly.graph_objects as go
    from dash import Dash, dcc, html
    from dash.dependencies import Input, Output
    # Create a new Dash app
    app = Dash(__name__)

    # Layout of the Dash app
    app.layout = html.Div([
        dcc.Graph(id='3d-scatter-plot'),
        html.Label("Opacity:"),
        dcc.Slider(
            id='opacity-slider',
            min=0,
            max=0.5,
            step=0.05,
            value=0.1,  # Default opacity value
        )
    ])

    # Define the callback to update the graph based on the slider value
    @app.callback(
        Output('3d-scatter-plot', 'figure'),
        Input('opacity-slider', 'value')
    )
    def update_figure(opacity_value):
        global image
        xyz_list = filter_by_opacity(image, opacity_threshold=opacity_value)

        # Combine all xyz data into a single NumPy array
        xyz_combined = np.vstack(xyz_list)  # Stack all xyz arrays vertically

        # Flatten xyz_combined for easier sampling and visualization
        xyz_combined_flattened = xyz_combined.reshape(-1, 3)

        # Sample a smaller number of points for visualization
        x_sample, y_sample, z_sample = sample_points(xyz_combined_flattened, sample_size=10000)

        # Create a 3D scatter plot with adjustable opacity
        fig = go.Figure(data=[go.Scatter3d(
            x=x_sample,
            y=y_sample,
            z=z_sample,
            mode='markers',
            marker=dict(size=2, color='blue')  # Apply opacity
        )])

        # Set axis labels and title
        fig.update_layout(
            scene=dict(
                xaxis_title='X Axis',
                yaxis_title='Y Axis',
                zaxis_title='Z Axis'
            ),
            title='3D Point Cloud Visualization with Opacity Control'
        )
        return fig

    return app


image = None
def main():
    global image
    # Load the image from a pickle file
    pickle_file_path = 'splatter_gt.pickle'  # Replace with your pickle file path
    image = load_image_from_pickle(pickle_file_path)

    xyz_list = filter_by_opacity(image, opacity_threshold=0.1)

    # Combine all xyz data into a single NumPy array
    xyz_combined = np.vstack(xyz_list)  # Stack all xyz arrays vertically

    # Flatten xyz_combined for easier sampling and visualization
    xyz_combined_flattened = xyz_combined.reshape(-1, 3)

    # Sample a smaller number of points for visualization
    x_sample, y_sample, z_sample = sample_points(xyz_combined_flattened, sample_size=10000)

    # Create and run the Dash app
    app = create_dash_app(x_sample, y_sample, z_sample)
    app.run_server(debug=True)


if __name__ == "__main__":
    main()