import streamlit as st 
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import tempfile
from scipy.ndimage import zoom
import os
import gdown
import traceback
import io

# --- Page Configuration ---
st.set_page_config(page_title="MRI Visualization and Segmentation", layout="wide")

# --- Model Configuration ---
MODEL_ID = '1r5EWxoBiCMF7ug6jly-3Oma4C9N4ZhGi'
MODEL_PATH = 'modelo_entrenado.pth'

# --- U-Net 2D Model Definition ---
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # Use bilinear upsampling or transposed convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=4, n_classes=3, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1) 

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# --- Helper Functions ---
@st.cache_data
def download_model_from_gdrive(model_id, model_path):
    try:
        gdown.download(f'https://drive.google.com/uc?id={model_id}', model_path, quiet=False)
        st.success(f"Model downloaded and saved to {model_path}")
    except Exception as e:
        st.error(f"Error downloading the model: {str(e)}")
    return model_path

def load_nifti1(file):
    if file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as temp_file:
            temp_file.write(file.read()) 
            temp_file.flush() 
            img = nib.load(temp_file.name) 
            return img.get_fdata()
    return None

# Function to display image slices
def plot_mri_slices1(data, modality):
    st.subheader(f"{modality} MRI")
    slice_idx = st.slider(f"Select an axial slice for {modality}", 0, data.shape[2] - 1, data.shape[2] // 2)
    plt.imshow(data[:, :, slice_idx], cmap='gray')
    plt.axis('off')
    st.pyplot(plt)
    
def load_nifti(file):
    if file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as temp_file:
                temp_file.write(file.read())
                temp_file.flush()
                img = nib.load(temp_file.name)
                return img.get_fdata(), img.shape 
        except Exception as e:
            st.error(f"Error loading NIfTI file: {str(e)}")
    return None, None

def preprocess_volume(volume, target_shape=(128, 128)):
    """
    Preprocesses a 4D (or 3D for single-channel) volume to be compatible with
    the 2D U-Net model.

    Args:
        volume (np.array): The volume to preprocess. Can be 3D (height, width, depth)
                           or 4D (height, width, depth, channels).
        target_shape (tuple): The desired shape for the height and width 
                             dimensions after preprocessing (default: (128, 128)).

    Returns:
        np.array: The preprocessed volume with shape (height, width, depth)
                  or (height, width, depth, channels), depending on input volume.
    """

    st.write(f"Checking dimensions of uploaded volume: {volume.shape}")

    # 1. Crop depth (if necessary)
    START_SLICE = 40 
    END_SLICE = 130 
    volume = volume[:, :, START_SLICE:END_SLICE]

    # 2. Resize spatial dimensions (if necessary)
    if volume.shape[0] != target_shape[0] or volume.shape[1] != target_shape[1]:
        st.write("Resizing volume...")
        factors = (
            target_shape[0] / volume.shape[0], 
            target_shape[1] / volume.shape[1],
            1, 
        ) 
        if len(volume.shape) == 4:  
            new_shape = (target_shape[0], target_shape[1], volume.shape[2], volume.shape[3])
            volume = zoom(volume, factors + (1,), order=1) 
        else:
            volume = zoom(volume, factors, order=1)
        st.write(f"New volume size after resizing: {volume.shape}")
    else:
        st.write("The volume already has the desired shape. It will not be resized.")

    # 3. Normalize 
    volume = (volume - volume.min()) / (volume.max() - volume.min())

    st.write(f"Volume Shape after preprocess_volume: {volume.shape}")
    return volume

def plot_mri_slices(data, modality, overlay=None):
    """Displays axial slices of a 3D volume, with an optional overlay."""
    st.subheader(f"{modality} MRI")

    if len(data.shape) < 3:
        st.error(f"Error: Expected at least 3 dimensions in image data, but found {len(data.shape)}")
        return

    slice_idx = st.slider(
        f"Select an axial slice for {modality}",
        0,
        data.shape[2] - 1,
        data.shape[2] // 2,
    )

    fig, ax = plt.subplots()
    ax.imshow(data[:, :, slice_idx], cmap="gray") 

    if overlay is not None:
        if overlay.shape[1:] != data.shape[:2]: 
            st.error(f"Error: The shapes of the image and the mask do not match: {data.shape} vs {overlay.shape}")
            return
        else:
            ax.imshow(overlay[0, :, :], cmap="hot", alpha=0.6)  

    ax.axis("off")
    st.pyplot(fig)

@st.cache_resource
def load_model():
    st.write("Loading the model...") 
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' does not exist. Downloading...")
        download_model_from_gdrive(MODEL_ID, MODEL_PATH)

    try:
        model = UNet(n_channels=4, n_classes=3) 
        st.write(f"Trying to load the model from {MODEL_PATH}...")
        state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        st.success("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.write(traceback.format_exc())
    return None

# --- Main Application Logic ---
if __name__ == "__main__":

    model = load_model()

    # Sidebar
    st.sidebar.title("Navigation")
    pagina = st.sidebar.radio(
        "Go to",
        [
            "MRI Visualization",
            "Segmentation Results",
            "Legends",
            "User Manual",
            "Surgical Planning",
        ],
    )

    # --- MRI Visualization Page ---
    if pagina == "MRI Visualization":
        st.title("MRI Visualization")
        st.write("Upload NIfTI files of different modalities to view slices.")

        t1_file = st.file_uploader("Upload the T1-weighted file (T1)", type=["nii", "nii.gz"])
        t1c_file = st.file_uploader("Upload the T1 file with contrast (T1c)", type=["nii", "nii.gz"])
        t2_file = st.file_uploader("Upload the T2-weighted (T2) file", type=["nii", "nii.gz"])
        flair_file = st.file_uploader("Upload the T2-FLAIR file", type=["nii", "nii.gz"])

        if t1_file:
            t1_data = load_nifti1(t1_file)
            if t1_data is not None:
                plot_mri_slices1(t1_data, "T1-weighted")

        if t1c_file:
            t1c_data = load_nifti1(t1c_file)
            if t1c_data is not None:
                plot_mri_slices1(t1c_data, "T1c (with contrast)")

        if t2_file:
            t2_data = load_nifti1(t2_file)
            if t2_data is not None:
                plot_mri_slices1(t2_data, "T2-weighted")

        if flair_file:
            flair_data = load_nifti1(flair_file)
            if flair_data is not None:
                plot_mri_slices1(flair_data, "T2-FLAIR")

    
# --- "Segmentation Results" Section ---
    elif pagina == "Segmentation Results":
        st.title("Segmentation Results")
        st.write("The results of the tumor segmentation will be shown here. Upload the stacked file (stack) to segment.")

        uploaded_stack = st.file_uploader(
            "Upload the stacked MRI file (.npy or .nii/.nii.gz)", 
            type=["npy", "nii", "nii.gz"]
        )

        if uploaded_stack is not None:
            try:
                # Loading Data
                if uploaded_stack.name.endswith('.npy'):
                    img_data = np.load(uploaded_stack)
                elif uploaded_stack.name.endswith(('.nii', '.nii.gz')):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as temp_file:
                        temp_file.write(uploaded_stack.read())
                        temp_file.flush() 
                        nii_img = nib.load(temp_file.name)
                        img_data = nii_img.get_fdata()  
                        st.write("NIfTI file loaded successfully.")
                    os.remove(temp_file.name) 
                else:
                    st.error("Unsupported file type. Please upload a .npy or .nii/.nii.gz file.")
                    st.stop()  

                # Dimension Checks
                if len(img_data.shape) != 4:
                    raise ValueError(f"Error: Expected 4 dimensions (height, width, depth, channels). Found: {img_data.shape}")

                # Preprocess Volume
                img_preprocessed = preprocess_volume(img_data)

                if img_preprocessed is not None and model is not None:
                    # --- Slider ---
                    slice_idx = st.slider(
                        "Select an axial slice to segment",
                        0,
                        img_preprocessed.shape[2] - 1,
                        img_preprocessed.shape[2] // 2,
                    )

                    with torch.no_grad():
                        img_slice = img_preprocessed[:, :, slice_idx, :]
                        img_tensor = torch.tensor(img_slice).unsqueeze(0).float()
                        img_tensor = img_tensor.permute(0, 3, 1, 2)
                        pred = model(img_tensor)
                        pred = torch.sigmoid(pred).squeeze(0).cpu().numpy() 

                    # --- Visualization ---
                    plot_mri_slices(img_preprocessed[:, :, :, 0], "Original T1", overlay=pred)

            except Exception as e:
                st.error(f"Error during segmentation: {e}")
                st.write(traceback.format_exc())
            
# --- Legend Page ---
    elif pagina == "Legends":
        st.title("Segmentation Legends")
        st.write(
            """
        In the segmented images, each value represents a type of tissue. Below is the legend to interpret the images:

        - 0: Background
        - 1: Necrotic tumor core (red)
        - 2: Enhanced Tumor (yellow)
        - 3: Peritumoral edematous tissue (green)
        """
        )

# --- User Manual Page ---
    elif pagina == "User Manual":
        st.title("User Manual")
        st.write(
            """
        MRI Viewer User Manual:

        1. Load Files: 
            - To Visualize: Upload MRI files in NIfTI format for each modality (T1, T2, T1c, FLAIR) on the "MRI Visualization" page. You can upload a single file that contains all modalities or each modality separately. 
            - For Segmentation: Upload a single file containing all 4 modalities (T1, T2, T1c, FLAIR) on the "Segmentation Results" page.
        2. Slice Viewing: Use the slider to select the axial slice you wish to view.
        3. Segmentation: Once you have uploaded a valid file, segmentation will run automatically and be displayed alongside the original image.
        4. Interpretation: Use the Legend page to understand the meaning of the colors in the segmentation.
        5. Surgical Planning: The "Surgical Planning" page provides information on how segmentation can assist in planning surgeries.
        """
        )

    # --- Surgical Planning Page ---
    elif pagina == "Surgical Planning":
        st.title("Applications in Surgical Planning")
        st.write(
            """
        Brain image segmentation plays a crucial role in planning surgeries for the resection of brain tumors. 
        By identifying the tumor core, edematous tissue, and enhanced tumor, surgeons can plan precise strategies for surgical intervention.

        This visualization and segmentation system allows physicians to:
        1. Observe the structure of the tumor in detail.
        2. Identify critical areas and risk zones.
        3. Plan the safest and most effective surgical route.
        4. Estimate tumor volume and extent of resection required.
        5. Assess tumor proximity to important brain structures.

        The accuracy of this information is vital for:
        - Maximizing tumor removal.
        - Minimizing damage to healthy brain tissue.
        - Improve the patient's postoperative outcomes.
        - Facilitate communication between the medical team and the patient.

        **Remember that this tool is an aid to clinical decision-making and should be used in conjunction with the experience of the neurosurgeon and other relevant clinical data.**
        """
        )

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("Developed by ACSI Group 1")
