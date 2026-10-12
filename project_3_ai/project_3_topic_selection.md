# INFO813 Project 3 — Topic Selection Guide

## Assignment Requirements Summary

| Requirement | Detail |
|---|---|
| **Course** | INFO813 Artificial Intelligence — T3, 2026 |
| **Weight** | 40% of total course marks |
| **Due** | 30 November 2026, 4:00 PM |
| **Group** | Max 2 members, topic must be approved by lecturer by end of Week 4 |
| **Domain** | **Image classification** using deep learning / ML |
| **Dataset** | ≥ 500 images, multiple classes, well-documented source |
| **Deliverables** | Report (Data Analysis → Problem Modelling → Software → Presentation) |

> [!IMPORTANT]
> The assignment brief explicitly states: *"Very simple problems or datasets that have been extensively covered during class activities should be avoided."*
> Your workspace already has a **Plant Disease Detection** system (PlantVillage dataset) — this topic is almost certainly **off-limits** since it appears to be a class example or prior project.

---

## Key Selection Criteria (from the brief)

1. **Well-structured image dataset** with multiple classes
2. **≥ 500 images** distributed across classes (larger preferred)
3. **Real-world application** preferred over artificial datasets
4. **Clear documentation** of source, class labels, and image characteristics
5. **Not too simple** — avoid topics extensively covered in class
6. Must support meaningful **data preprocessing, model evaluation, and software demo**

---

## Recommended Topics

### 🏆 Tier 1: Strong Choices (Rich datasets, clear real-world impact, lots of literature)

| # | Topic | Dataset (Kaggle/UCI) | Classes | Images | Why It's Strong |
|---|---|---|---|---|---|
| **1** | **Skin Lesion / Dermatology Classification** | [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) or [ISIC Archive](https://www.isic-archive.com/) | 7 skin lesion types (melanoma, basal cell carcinoma, etc.) | ~10,000+ | Medical AI is high-impact; strong class imbalance challenge; rich preprocessing (hair removal, augmentation); excellent academic literature. Directly ties into AI ethics discussion. |
| **2** | **Chest X-Ray Disease Detection** | [ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data) or [COVID-19 Radiography](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) | 3-4 (Normal, COVID, Pneumonia, Lung Opacity) | ~21,000+ | Topical, strong transfer learning application, well-documented. Multi-label variant adds complexity. |
| **3** | **Waste / Recycling Classification** | [TrashNet](https://www.kaggle.com/datasets/fedesoriano/garbage-classification) or [Waste Classification](https://www.kaggle.com/datasets/techsash/waste-classification-data) | 6-12 (cardboard, glass, metal, paper, plastic, trash) | ~15,000+ | Environmental sustainability angle; practical deployment potential (smart bins); good variety of textures and materials. |
| **4** | **Wildlife / Animal Species Recognition** | [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) or [iNaturalist subset](https://www.kaggle.com/competitions/inaturalist-2021) | 10+ species | ~26,000+ | Conservation application; fine-grained classification challenge; good for demonstrating data augmentation. |
| **5** | **Food Recognition / Classification** | [Food-101](https://www.kaggle.com/datasets/dansbecker/food-101) or [Food-11](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset) | 11-101 categories | ~16,000+ | Practical (nutrition tracking apps); high inter-class similarity challenge; large well-curated dataset. |

### 🥈 Tier 2: Solid Choices (Good datasets, slightly narrower scope)

| # | Topic | Dataset | Classes | Images | Notes |
|---|---|---|---|---|---|
| **6** | **Traffic Sign Recognition** | [GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) | 43 | ~50,000+ | Autonomous driving application; mentioned in the brief as an example; varying lighting/angle challenges. |
| **7** | **Malaria Cell Detection** | [Malaria Cell Images](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) | 2 (parasitized/uninfected) | ~27,000 | Medical impact; binary classification is simpler but can be extended with severity grading. Only 2 classes is a slight weakness. |
| **8** | **Architectural Style Classification** | [Architectural Styles](https://www.kaggle.com/datasets/dumitrux/architectural-styles-dataset) | 10+ styles | ~5,000+ | Cultural/heritage application; subtle visual differences between styles add complexity. |
| **9** | **Crop Disease Detection (non-PlantVillage)** | [Cassava Leaf Disease](https://www.kaggle.com/competitions/cassava-leaf-disease-classification) | 5 | ~21,000 | Agriculture focus without overlapping with class material; real Kaggle competition dataset. |
| **10** | **Facial Expression Recognition** | [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) | 7 emotions | ~35,000 | HCI/mental health application; grayscale images add preprocessing discussion. Privacy/ethics angle. |

---

## My Top 3 Recommendations

### 🥇 1. Skin Lesion Classification (HAM10000)
**Why:** Ticks every box — medical AI has massive real-world impact, the dataset is well-documented and cited in hundreds of papers, has meaningful class imbalance (great for evaluation discussion), requires sophisticated preprocessing (image normalization, augmentation, handling of artifacts like hair/rulers), and transfer learning with EfficientNet/ResNet is well-studied. You can build an impressive interactive tool ("upload a skin photo → get risk assessment"). Strong ethical discussion potential too.

### 🥈 2. Waste / Recycling Classification
**Why:** Sustainability is a compelling narrative. The dataset has good class variety (6-12 categories), practical deployment potential (smart recycling bins), and the images have diverse backgrounds and lighting — great for demonstrating robust preprocessing. The software demo (upload waste photo → classification + recycling guidance) is intuitive and impressive.

### 🥉 3. Chest X-Ray Disease Detection
**Why:** Extremely topical (post-pandemic relevance), excellent datasets, and strong academic backing. The binary/multi-class nature allows you to start simple and extend. Grad-CAM visualizations on X-rays are particularly compelling for the presentation.

---

## What to Avoid

| ❌ Avoid | Why |
|---|---|
| Plant Disease Detection (PlantVillage) | Already built in this workspace — likely a class example |
| MNIST / Fashion-MNIST / CIFAR-10 | Too simple, extensively covered in class |
| Cats vs Dogs | Too simple, overused |
| Handwriting recognition | Likely covered in class |
| Datasets with < 500 images | Below the minimum requirement |
| Artificially generated datasets | Brief explicitly prefers real-world data |

---

## Next Steps

1. **Pick a topic** from the recommendations above (or propose your own)
2. **Verify the dataset** — download and check class distribution, image quality
3. **Get lecturer approval** (required by end of Week 4)
4. I can then help you build a full project plan, set up the codebase, and start the data analysis
