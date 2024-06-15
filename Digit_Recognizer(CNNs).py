{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyM/h4VCSEOzPgkzrKncU0Nh"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import accuracy_score\n",
        "\n",
        "import tensorflow as tf\n",
        "from tensorflow.keras.layers import Conv2D, MaxPool2D, ZeroPadding2D, ReLU, Flatten,Dense, BatchNormalization,AveragePooling2D\n",
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.losses import SparseCategoricalCrossentropy\n",
        "from tensorflow.keras.preprocessing.image import ImageDataGenerator"
      ],
      "metadata": {
        "id": "gD7LbosNgSOF"
      },
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "files = os.listdir('/content/drive/My Drive')\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "JHrlZYb2FCDf",
        "outputId": "7b47e2f7-8bb6-437a-a7d8-5aa07fdd0d09"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "file_path1='/content/drive/My Drive/Colab Notebooks/Kaggle notebooks/train.csv'\n",
        "file_path2='/content/drive/My Drive/Colab Notebooks/Kaggle notebooks/test.csv'"
      ],
      "metadata": {
        "id": "jIxN4Wt-FQ7k"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "id": "KWA6qAU0cVHs"
      },
      "outputs": [],
      "source": [
        "d_train=pd.read_csv(file_path1)\n",
        "d_test=pd.read_csv(file_path2)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "d_test['train']=0\n",
        "d_train['train']=1\n",
        "d=pd.concat((d_test,d_train), axis=0)"
      ],
      "metadata": {
        "id": "dpdEBPGq2wkr"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(d.shape)"
      ],
      "metadata": {
        "id": "4clCCbuse0a0",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "003944f2-039a-4129-8370-798a78a45a89"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(70000, 786)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "2n8GTICv2YNx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "#data visualization"
      ],
      "metadata": {
        "id": "d3dcA6VU2egn"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "d.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 255
        },
        "id": "zSBnDeLx2iDZ",
        "outputId": "1e50b9c3-d6e1-451b-ef34-86f79de98032"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   pixel0  pixel1  pixel2  pixel3  pixel4  pixel5  pixel6  pixel7  pixel8  \\\n",
              "0       0       0       0       0       0       0       0       0       0   \n",
              "1       0       0       0       0       0       0       0       0       0   \n",
              "2       0       0       0       0       0       0       0       0       0   \n",
              "3       0       0       0       0       0       0       0       0       0   \n",
              "4       0       0       0       0       0       0       0       0       0   \n",
              "\n",
              "   pixel9  ...  pixel776  pixel777  pixel778  pixel779  pixel780  pixel781  \\\n",
              "0       0  ...         0         0         0         0         0         0   \n",
              "1       0  ...         0         0         0         0         0         0   \n",
              "2       0  ...         0         0         0         0         0         0   \n",
              "3       0  ...         0         0         0         0         0         0   \n",
              "4       0  ...         0         0         0         0         0         0   \n",
              "\n",
              "   pixel782  pixel783  train  label  \n",
              "0         0         0      0    NaN  \n",
              "1         0         0      0    NaN  \n",
              "2         0         0      0    NaN  \n",
              "3         0         0      0    NaN  \n",
              "4         0         0      0    NaN  \n",
              "\n",
              "[5 rows x 786 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-78fdc72f-345f-480c-aa39-756f3fa3d18a\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>pixel0</th>\n",
              "      <th>pixel1</th>\n",
              "      <th>pixel2</th>\n",
              "      <th>pixel3</th>\n",
              "      <th>pixel4</th>\n",
              "      <th>pixel5</th>\n",
              "      <th>pixel6</th>\n",
              "      <th>pixel7</th>\n",
              "      <th>pixel8</th>\n",
              "      <th>pixel9</th>\n",
              "      <th>...</th>\n",
              "      <th>pixel776</th>\n",
              "      <th>pixel777</th>\n",
              "      <th>pixel778</th>\n",
              "      <th>pixel779</th>\n",
              "      <th>pixel780</th>\n",
              "      <th>pixel781</th>\n",
              "      <th>pixel782</th>\n",
              "      <th>pixel783</th>\n",
              "      <th>train</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5 rows × 786 columns</p>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-78fdc72f-345f-480c-aa39-756f3fa3d18a')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-78fdc72f-345f-480c-aa39-756f3fa3d18a button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-78fdc72f-345f-480c-aa39-756f3fa3d18a');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-a1cfd91d-20bf-4111-9c53-ec491f8e99af\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-a1cfd91d-20bf-4111-9c53-ec491f8e99af')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-a1cfd91d-20bf-4111-9c53-ec491f8e99af button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "d"
            }
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "d.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "EOGRuejU2jwp",
        "outputId": "08aa7c5b-17b1-4b3b-a809-4df3cdd6c756"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "Index: 70000 entries, 0 to 41999\n",
            "Columns: 786 entries, pixel0 to label\n",
            "dtypes: float64(1), int64(785)\n",
            "memory usage: 420.3 MB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "d.describe()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 349
        },
        "id": "5ag6yyy75PnI",
        "outputId": "c55bb6ab-76cb-479c-928e-e33bd0d9a010"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "        pixel0   pixel1   pixel2   pixel3   pixel4   pixel5   pixel6   pixel7  \\\n",
              "count  70000.0  70000.0  70000.0  70000.0  70000.0  70000.0  70000.0  70000.0   \n",
              "mean       0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0   \n",
              "std        0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0   \n",
              "min        0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0   \n",
              "25%        0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0   \n",
              "50%        0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0   \n",
              "75%        0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0   \n",
              "max        0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0   \n",
              "\n",
              "        pixel8   pixel9  ...      pixel776      pixel777      pixel778  \\\n",
              "count  70000.0  70000.0  ...  70000.000000  70000.000000  70000.000000   \n",
              "mean       0.0      0.0  ...      0.046629      0.016614      0.012957   \n",
              "std        0.0      0.0  ...      2.783732      1.561822      1.553796   \n",
              "min        0.0      0.0  ...      0.000000      0.000000      0.000000   \n",
              "25%        0.0      0.0  ...      0.000000      0.000000      0.000000   \n",
              "50%        0.0      0.0  ...      0.000000      0.000000      0.000000   \n",
              "75%        0.0      0.0  ...      0.000000      0.000000      0.000000   \n",
              "max        0.0      0.0  ...    253.000000    253.000000    254.000000   \n",
              "\n",
              "           pixel779  pixel780  pixel781  pixel782  pixel783         train  \\\n",
              "count  70000.000000   70000.0   70000.0   70000.0   70000.0  70000.000000   \n",
              "mean       0.001714       0.0       0.0       0.0       0.0      0.600000   \n",
              "std        0.320889       0.0       0.0       0.0       0.0      0.489901   \n",
              "min        0.000000       0.0       0.0       0.0       0.0      0.000000   \n",
              "25%        0.000000       0.0       0.0       0.0       0.0      0.000000   \n",
              "50%        0.000000       0.0       0.0       0.0       0.0      1.000000   \n",
              "75%        0.000000       0.0       0.0       0.0       0.0      1.000000   \n",
              "max       62.000000       0.0       0.0       0.0       0.0      1.000000   \n",
              "\n",
              "              label  \n",
              "count  42000.000000  \n",
              "mean       4.456643  \n",
              "std        2.887730  \n",
              "min        0.000000  \n",
              "25%        2.000000  \n",
              "50%        4.000000  \n",
              "75%        7.000000  \n",
              "max        9.000000  \n",
              "\n",
              "[8 rows x 786 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-67104722-623b-497e-a690-53f5da35c271\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>pixel0</th>\n",
              "      <th>pixel1</th>\n",
              "      <th>pixel2</th>\n",
              "      <th>pixel3</th>\n",
              "      <th>pixel4</th>\n",
              "      <th>pixel5</th>\n",
              "      <th>pixel6</th>\n",
              "      <th>pixel7</th>\n",
              "      <th>pixel8</th>\n",
              "      <th>pixel9</th>\n",
              "      <th>...</th>\n",
              "      <th>pixel776</th>\n",
              "      <th>pixel777</th>\n",
              "      <th>pixel778</th>\n",
              "      <th>pixel779</th>\n",
              "      <th>pixel780</th>\n",
              "      <th>pixel781</th>\n",
              "      <th>pixel782</th>\n",
              "      <th>pixel783</th>\n",
              "      <th>train</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>70000.0</td>\n",
              "      <td>70000.0</td>\n",
              "      <td>70000.0</td>\n",
              "      <td>70000.0</td>\n",
              "      <td>70000.0</td>\n",
              "      <td>70000.0</td>\n",
              "      <td>70000.0</td>\n",
              "      <td>70000.0</td>\n",
              "      <td>70000.0</td>\n",
              "      <td>70000.0</td>\n",
              "      <td>...</td>\n",
              "      <td>70000.000000</td>\n",
              "      <td>70000.000000</td>\n",
              "      <td>70000.000000</td>\n",
              "      <td>70000.000000</td>\n",
              "      <td>70000.0</td>\n",
              "      <td>70000.0</td>\n",
              "      <td>70000.0</td>\n",
              "      <td>70000.0</td>\n",
              "      <td>70000.000000</td>\n",
              "      <td>42000.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>...</td>\n",
              "      <td>0.046629</td>\n",
              "      <td>0.016614</td>\n",
              "      <td>0.012957</td>\n",
              "      <td>0.001714</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.600000</td>\n",
              "      <td>4.456643</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>...</td>\n",
              "      <td>2.783732</td>\n",
              "      <td>1.561822</td>\n",
              "      <td>1.553796</td>\n",
              "      <td>0.320889</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.489901</td>\n",
              "      <td>2.887730</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>...</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>...</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>2.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>...</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>4.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>...</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>7.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>...</td>\n",
              "      <td>253.000000</td>\n",
              "      <td>253.000000</td>\n",
              "      <td>254.000000</td>\n",
              "      <td>62.000000</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>9.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>8 rows × 786 columns</p>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-67104722-623b-497e-a690-53f5da35c271')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-67104722-623b-497e-a690-53f5da35c271 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-67104722-623b-497e-a690-53f5da35c271');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-5d217d1c-7283-4d84-b9b7-33d353b05c03\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-5d217d1c-7283-4d84-b9b7-33d353b05c03')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-5d217d1c-7283-4d84-b9b7-33d353b05c03 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe"
            }
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(d[d['train']==0].shape[0])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "96ceL4Di5eK_",
        "outputId": "8884cdb6-ffb1-426a-f59d-63e1b06ca3cc"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "28000\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#preprocessing"
      ],
      "metadata": {
        "id": "wD9IYdG965nm"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#standard scaling\n",
        "columns_to_scale=list()\n",
        "scaler=StandardScaler()\n",
        "for col in d.columns:\n",
        "  if col not in ['train','label']:\n",
        "    columns_to_scale.append(col)\n",
        "\n",
        "d[columns_to_scale]=scaler.fit_transform(d[columns_to_scale])"
      ],
      "metadata": {
        "id": "w_9jeHcw66-g"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(d_train.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2B9iuWaz751G",
        "outputId": "7bb2288f-4e4c-413a-e3f6-07daf3b6c933"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(42000, 786)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "d_train=d[d['train']==1]\n",
        "d_test=d[d['train']==0]\n",
        "d_test=d_test.drop(columns=['label', 'train'])\n",
        "x=d_train.drop(columns=['label','train'])\n",
        "y=d_train['label']\n",
        "\n",
        "x=np.array(x)\n",
        "y=np.array(y)\n",
        "d_test=np.array(d_test)\n",
        "x=x.reshape(-1,28,28,1)\n",
        "y=y.reshape(-1,1)\n",
        "d_test=d_test.reshape(-1,28,28,1)\n"
      ],
      "metadata": {
        "id": "527cf6De8BGG"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(x_train.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 141
        },
        "id": "dHPTCDuBfFMB",
        "outputId": "f8df6979-bae5-461a-ab17-24879994bf7b"
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "error",
          "ename": "NameError",
          "evalue": "name 'x_train' is not defined",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-11-b13f4b81d166>\u001b[0m in \u001b[0;36m<cell line: 1>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx_train\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m: name 'x_train' is not defined"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "x_train, x_ct, y_train, y_ct= train_test_split(x,y, train_size=0.6)\n",
        "x_cv, x_test, y_cv, y_test=train_test_split(x_ct, y_ct, train_size=0.5)"
      ],
      "metadata": {
        "id": "sn91Xisdd9Py"
      },
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "ya3n0_Fo4xHk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "datagen=ImageDataGenerator(\n",
        "    rotation_range=5,\n",
        "    width_shift_range=0.1,\n",
        "    height_shift_range=0.1,\n",
        "    shear_range=0.2,\n",
        "    zoom_range=0.2,\n",
        "    horizontal_flip=True,\n",
        "    vertical_flip=True\n",
        ")\n",
        "\n",
        "datagen.fit(x_train)\n",
        "\n"
      ],
      "metadata": {
        "id": "UKdODgR6q6ze"
      },
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_gen=datagen.flow(x_train,y_train,batch_size=32)\n"
      ],
      "metadata": {
        "id": "PtGQXGDBr5uC"
      },
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x_traincv=np.concatenate((x_train,x_cv), axis=0)\n",
        "y_traincv=np.concatenate((y_train,y_cv), axis=0)"
      ],
      "metadata": {
        "id": "1Xvr4gdPmmJN"
      },
      "execution_count": 17,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(x_train.shape, y_train.shape, d_test.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RJ1XxdnBNZWu",
        "outputId": "d0f46f5e-427e-4826-854d-45e5fc6c8a1b"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(25200, 28, 28, 1) (25200, 1) (28000, 28, 28, 1)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(y_test[9])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "iUKtRbJVCE3o",
        "outputId": "94b3a463-ee72-4738-cdb9-44a95f29c774"
      },
      "execution_count": 35,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[8.]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#convolutional neural network"
      ],
      "metadata": {
        "id": "CV84X-Kpcl2f"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "print(x_train.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8sYoCYJOevQW",
        "outputId": "b4724c12-1b27-494b-8881-2a79cb201f3b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(25200, 28, 28, 1)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#cnn using sequential api\n",
        "cnn_model=Sequential([\n",
        "    Conv2D(filters=100, kernel_size=(5,5), input_shape=(28,28,1), padding='same', activation='relu'),\n",
        "    MaxPool2D(pool_size=(5,5)),\n",
        "    BatchNormalization(),\n",
        "    Conv2D(filters=50, kernel_size=(3,3),padding='same', activation='relu'),\n",
        "    MaxPool2D(pool_size=(3,3)),\n",
        "\n",
        "    Flatten(),\n",
        "    BatchNormalization(),\n",
        "    Dense(50,activation='relu'),\n",
        "    Dense(10, activation='softmax')\n",
        "])\n",
        "\n",
        "cnn_model.compile(\n",
        "    loss=tf.keras.losses.SparseCategoricalCrossentropy(),\n",
        "    optimizer=tf.keras.optimizers.Adam(0.009),\n",
        "    metrics=['accuracy']\n",
        ")"
      ],
      "metadata": {
        "id": "aD5eQiV3cooS"
      },
      "execution_count": 19,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn_model.fit(x_train, y_train, epochs=30, validation_data=(x_cv,y_cv))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "y8GBKxBUMeaX",
        "outputId": "e630353e-95e4-4698-d7b9-b5d318a2d088"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/30\n",
            "788/788 [==============================] - 10s 11ms/step - loss: 0.2886 - accuracy: 0.9137 - val_loss: 0.1822 - val_accuracy: 0.9469\n",
            "Epoch 2/30\n",
            "788/788 [==============================] - 8s 11ms/step - loss: 0.1422 - accuracy: 0.9578 - val_loss: 0.1647 - val_accuracy: 0.9502\n",
            "Epoch 3/30\n",
            "788/788 [==============================] - 8s 10ms/step - loss: 0.1170 - accuracy: 0.9647 - val_loss: 0.1122 - val_accuracy: 0.9688\n",
            "Epoch 4/30\n",
            "788/788 [==============================] - 8s 10ms/step - loss: 0.0979 - accuracy: 0.9694 - val_loss: 0.1114 - val_accuracy: 0.9686\n",
            "Epoch 5/30\n",
            "788/788 [==============================] - 8s 10ms/step - loss: 0.0899 - accuracy: 0.9730 - val_loss: 0.0878 - val_accuracy: 0.9738\n",
            "Epoch 6/30\n",
            "788/788 [==============================] - 8s 10ms/step - loss: 0.0853 - accuracy: 0.9725 - val_loss: 0.1111 - val_accuracy: 0.9724\n",
            "Epoch 7/30\n",
            "788/788 [==============================] - 8s 10ms/step - loss: 0.0822 - accuracy: 0.9753 - val_loss: 0.0910 - val_accuracy: 0.9750\n",
            "Epoch 8/30\n",
            "788/788 [==============================] - 8s 10ms/step - loss: 0.0731 - accuracy: 0.9770 - val_loss: 0.1609 - val_accuracy: 0.9631\n",
            "Epoch 9/30\n",
            "788/788 [==============================] - 8s 10ms/step - loss: 0.0689 - accuracy: 0.9799 - val_loss: 0.1029 - val_accuracy: 0.9770\n",
            "Epoch 10/30\n",
            "788/788 [==============================] - 8s 10ms/step - loss: 0.0639 - accuracy: 0.9811 - val_loss: 0.1193 - val_accuracy: 0.9680\n",
            "Epoch 11/30\n",
            "788/788 [==============================] - 8s 10ms/step - loss: 0.0616 - accuracy: 0.9819 - val_loss: 0.1391 - val_accuracy: 0.9668\n",
            "Epoch 12/30\n",
            "788/788 [==============================] - 8s 10ms/step - loss: 0.0571 - accuracy: 0.9821 - val_loss: 0.0996 - val_accuracy: 0.9750\n",
            "Epoch 13/30\n",
            "788/788 [==============================] - 8s 10ms/step - loss: 0.0616 - accuracy: 0.9808 - val_loss: 0.1629 - val_accuracy: 0.9589\n",
            "Epoch 14/30\n",
            "788/788 [==============================] - 8s 10ms/step - loss: 0.0520 - accuracy: 0.9844 - val_loss: 0.0976 - val_accuracy: 0.9781\n",
            "Epoch 15/30\n",
            "788/788 [==============================] - 8s 10ms/step - loss: 0.0530 - accuracy: 0.9841 - val_loss: 0.1184 - val_accuracy: 0.9765\n",
            "Epoch 16/30\n",
            "788/788 [==============================] - 8s 10ms/step - loss: 0.0507 - accuracy: 0.9843 - val_loss: 0.1062 - val_accuracy: 0.9765\n",
            "Epoch 17/30\n",
            "788/788 [==============================] - 8s 10ms/step - loss: 0.0502 - accuracy: 0.9851 - val_loss: 0.1272 - val_accuracy: 0.9737\n",
            "Epoch 18/30\n",
            "788/788 [==============================] - 8s 10ms/step - loss: 0.0417 - accuracy: 0.9874 - val_loss: 0.1757 - val_accuracy: 0.9617\n",
            "Epoch 19/30\n",
            "788/788 [==============================] - 8s 10ms/step - loss: 0.0438 - accuracy: 0.9866 - val_loss: 0.0928 - val_accuracy: 0.9802\n",
            "Epoch 20/30\n",
            "788/788 [==============================] - 8s 10ms/step - loss: 0.0455 - accuracy: 0.9861 - val_loss: 0.1136 - val_accuracy: 0.9736\n",
            "Epoch 21/30\n",
            "788/788 [==============================] - 8s 10ms/step - loss: 0.0475 - accuracy: 0.9865 - val_loss: 0.1092 - val_accuracy: 0.9756\n",
            "Epoch 22/30\n",
            "788/788 [==============================] - 8s 10ms/step - loss: 0.0459 - accuracy: 0.9867 - val_loss: 0.0972 - val_accuracy: 0.9776\n",
            "Epoch 23/30\n",
            "788/788 [==============================] - 8s 10ms/step - loss: 0.0391 - accuracy: 0.9875 - val_loss: 0.1181 - val_accuracy: 0.9708\n",
            "Epoch 24/30\n",
            "788/788 [==============================] - 8s 10ms/step - loss: 0.0426 - accuracy: 0.9868 - val_loss: 0.2640 - val_accuracy: 0.9713\n",
            "Epoch 25/30\n",
            "788/788 [==============================] - 8s 10ms/step - loss: 0.0446 - accuracy: 0.9870 - val_loss: 0.1831 - val_accuracy: 0.9735\n",
            "Epoch 26/30\n",
            "788/788 [==============================] - 8s 10ms/step - loss: 0.0383 - accuracy: 0.9888 - val_loss: 0.1088 - val_accuracy: 0.9798\n",
            "Epoch 27/30\n",
            "788/788 [==============================] - 8s 10ms/step - loss: 0.0391 - accuracy: 0.9879 - val_loss: 0.1141 - val_accuracy: 0.9780\n",
            "Epoch 28/30\n",
            "788/788 [==============================] - 8s 10ms/step - loss: 0.0369 - accuracy: 0.9894 - val_loss: 0.1280 - val_accuracy: 0.9788\n",
            "Epoch 29/30\n",
            "788/788 [==============================] - 8s 10ms/step - loss: 0.0389 - accuracy: 0.9888 - val_loss: 0.1202 - val_accuracy: 0.9796\n",
            "Epoch 30/30\n",
            "788/788 [==============================] - 8s 10ms/step - loss: 0.0392 - accuracy: 0.9888 - val_loss: 0.1230 - val_accuracy: 0.9743\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.src.callbacks.History at 0x79f54d34cdf0>"
            ]
          },
          "metadata": {},
          "execution_count": 20
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cnn_model.summary()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fwlITd6Bx1kQ",
        "outputId": "d49ddf3e-82d8-4ea5-f6d6-dcec13a3c8a1"
      },
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"sequential\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " conv2d (Conv2D)             (None, 28, 28, 100)       2600      \n",
            "                                                                 \n",
            " max_pooling2d (MaxPooling2  (None, 5, 5, 100)         0         \n",
            " D)                                                              \n",
            "                                                                 \n",
            " batch_normalization (Batch  (None, 5, 5, 100)         400       \n",
            " Normalization)                                                  \n",
            "                                                                 \n",
            " conv2d_1 (Conv2D)           (None, 5, 5, 50)          45050     \n",
            "                                                                 \n",
            " max_pooling2d_1 (MaxPoolin  (None, 1, 1, 50)          0         \n",
            " g2D)                                                            \n",
            "                                                                 \n",
            " flatten (Flatten)           (None, 50)                0         \n",
            "                                                                 \n",
            " batch_normalization_1 (Bat  (None, 50)                200       \n",
            " chNormalization)                                                \n",
            "                                                                 \n",
            " dense (Dense)               (None, 50)                2550      \n",
            "                                                                 \n",
            " dense_1 (Dense)             (None, 10)                510       \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 51310 (200.43 KB)\n",
            "Trainable params: 51010 (199.26 KB)\n",
            "Non-trainable params: 300 (1.17 KB)\n",
            "_________________________________________________________________\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "predictions_train=cnn_model.predict(x_train)\n",
        "predictions_train=np.argmax(predictions_train, axis=1)\n",
        "accuracy_train=accuracy_score(predictions_train, y_train)\n",
        "print(f\"accuracy score for train set is {accuracy_train}\")\n",
        "\n",
        "\n",
        "predictions_test=cnn_model.predict(x_test)\n",
        "predictions_test=np.argmax(predictions_test,axis=1)\n",
        "accuracy_test=accuracy_score(predictions_test, y_test)\n",
        "print(f\"accuracy score for test set is {accuracy_test}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "V0YdQJSm3OgB",
        "outputId": "e762ba66-2198-4b00-e196-4baf3f710ef3"
      },
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "788/788 [==============================] - 13s 16ms/step\n",
            "accuracy score for train set is 0.10531746031746032\n",
            "263/263 [==============================] - 3s 12ms/step\n",
            "accuracy score for test set is 0.10809523809523809\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(x_train.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "uGrY-DCo6-gk",
        "outputId": "3eb4b589-0da1-4789-ad8e-98a08483773b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(25200, 28, 28, 1)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "cnn using functional api"
      ],
      "metadata": {
        "id": "BIecb42bM_Q8"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#cnn using functional api\n",
        "\n",
        "input_img=tf.keras.Input(shape=(28,28,1) )\n",
        "a1=Conv2D(filters=100, kernel_size=(5,5),padding='same', activation='relu')(input_img)\n",
        "p1=MaxPool2D(pool_size=(5,5))(a1)\n",
        "p1=BatchNormalization()(p1)\n",
        "z2=Conv2D(filters=50, kernel_size=(3,3), padding='same')(p1)\n",
        "a2=ReLU()(z2)\n",
        "p2=MaxPool2D(pool_size=(3,3))(a2)\n",
        "p2=BatchNormalization()(p2)\n",
        "\n",
        "p2=Flatten()(p2)\n",
        "\n",
        "a3=Dense(50,activation='relu')(p2)\n",
        "outputs=Dense(10,activation='softmax')(a3)\n",
        "\n"
      ],
      "metadata": {
        "id": "h8bj-7LOM9ql"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn_model=tf.keras.Model(inputs=input_img, outputs=outputs)\n",
        "\n",
        "cnn_model.compile(\n",
        "    loss=tf.keras.losses.SparseCategoricalCrossentropy(),\n",
        "    optimizer=tf.keras.optimizers.Adam(0.009),\n",
        "    metrics=['accuracy']\n",
        ")\n",
        "\n",
        "cnn_model.fit(train_gen, epochs=150, validation_data=(x_cv,y_cv))\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FrdiTpGhR6Gw",
        "outputId": "0ffff663-7c7e-493b-ec1a-d9c5d8c8e090"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/150\n",
            "788/788 [==============================] - 13s 15ms/step - loss: 0.2357 - accuracy: 0.0989 - val_loss: 0.1763 - val_accuracy: 0.1040\n",
            "Epoch 2/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2417 - accuracy: 0.0987 - val_loss: 0.1376 - val_accuracy: 0.0988\n",
            "Epoch 3/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2345 - accuracy: 0.0987 - val_loss: 0.1536 - val_accuracy: 0.1020\n",
            "Epoch 4/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2363 - accuracy: 0.0989 - val_loss: 0.2012 - val_accuracy: 0.1011\n",
            "Epoch 5/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2234 - accuracy: 0.0983 - val_loss: 0.1498 - val_accuracy: 0.1019\n",
            "Epoch 6/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2291 - accuracy: 0.0988 - val_loss: 0.1625 - val_accuracy: 0.1015\n",
            "Epoch 7/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2188 - accuracy: 0.0985 - val_loss: 0.1762 - val_accuracy: 0.0987\n",
            "Epoch 8/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2262 - accuracy: 0.0981 - val_loss: 0.1604 - val_accuracy: 0.1005\n",
            "Epoch 9/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2234 - accuracy: 0.0987 - val_loss: 0.1501 - val_accuracy: 0.0962\n",
            "Epoch 10/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2184 - accuracy: 0.0989 - val_loss: 0.1476 - val_accuracy: 0.0999\n",
            "Epoch 11/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2251 - accuracy: 0.0988 - val_loss: 0.1240 - val_accuracy: 0.0987\n",
            "Epoch 12/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2130 - accuracy: 0.0984 - val_loss: 0.1809 - val_accuracy: 0.0971\n",
            "Epoch 13/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2142 - accuracy: 0.0988 - val_loss: 0.1585 - val_accuracy: 0.0967\n",
            "Epoch 14/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2061 - accuracy: 0.0988 - val_loss: 0.1424 - val_accuracy: 0.1000\n",
            "Epoch 15/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2060 - accuracy: 0.0984 - val_loss: 0.1708 - val_accuracy: 0.1015\n",
            "Epoch 16/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2158 - accuracy: 0.0991 - val_loss: 0.1582 - val_accuracy: 0.1010\n",
            "Epoch 17/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2066 - accuracy: 0.0987 - val_loss: 0.1737 - val_accuracy: 0.0996\n",
            "Epoch 18/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2061 - accuracy: 0.0982 - val_loss: 0.1339 - val_accuracy: 0.1013\n",
            "Epoch 19/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2076 - accuracy: 0.0981 - val_loss: 0.1743 - val_accuracy: 0.1006\n",
            "Epoch 20/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2065 - accuracy: 0.0988 - val_loss: 0.1571 - val_accuracy: 0.1018\n",
            "Epoch 21/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2159 - accuracy: 0.0985 - val_loss: 0.1268 - val_accuracy: 0.1000\n",
            "Epoch 22/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2061 - accuracy: 0.0987 - val_loss: 0.1207 - val_accuracy: 0.1027\n",
            "Epoch 23/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2010 - accuracy: 0.0989 - val_loss: 0.1313 - val_accuracy: 0.1026\n",
            "Epoch 24/150\n",
            "788/788 [==============================] - 11s 15ms/step - loss: 0.2164 - accuracy: 0.0985 - val_loss: 0.1473 - val_accuracy: 0.1014\n",
            "Epoch 25/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2027 - accuracy: 0.0989 - val_loss: 0.1986 - val_accuracy: 0.0987\n",
            "Epoch 26/150\n",
            "788/788 [==============================] - 11s 15ms/step - loss: 0.2087 - accuracy: 0.0985 - val_loss: 0.1450 - val_accuracy: 0.0994\n",
            "Epoch 27/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2005 - accuracy: 0.0979 - val_loss: 0.1401 - val_accuracy: 0.0986\n",
            "Epoch 28/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2034 - accuracy: 0.0990 - val_loss: 0.1325 - val_accuracy: 0.1014\n",
            "Epoch 29/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1989 - accuracy: 0.0984 - val_loss: 0.1267 - val_accuracy: 0.1021\n",
            "Epoch 30/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1975 - accuracy: 0.0987 - val_loss: 0.1735 - val_accuracy: 0.1007\n",
            "Epoch 31/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1992 - accuracy: 0.0988 - val_loss: 0.1319 - val_accuracy: 0.1036\n",
            "Epoch 32/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2049 - accuracy: 0.0987 - val_loss: 0.1610 - val_accuracy: 0.1005\n",
            "Epoch 33/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1975 - accuracy: 0.0985 - val_loss: 0.1557 - val_accuracy: 0.1007\n",
            "Epoch 34/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2015 - accuracy: 0.0982 - val_loss: 0.2246 - val_accuracy: 0.1011\n",
            "Epoch 35/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1954 - accuracy: 0.0981 - val_loss: 0.1683 - val_accuracy: 0.1001\n",
            "Epoch 36/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1951 - accuracy: 0.0986 - val_loss: 0.1723 - val_accuracy: 0.1001\n",
            "Epoch 37/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2011 - accuracy: 0.0987 - val_loss: 0.1880 - val_accuracy: 0.0967\n",
            "Epoch 38/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1980 - accuracy: 0.0986 - val_loss: 0.1331 - val_accuracy: 0.1001\n",
            "Epoch 39/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1923 - accuracy: 0.0989 - val_loss: 0.1561 - val_accuracy: 0.0983\n",
            "Epoch 40/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2027 - accuracy: 0.0981 - val_loss: 0.1675 - val_accuracy: 0.1008\n",
            "Epoch 41/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1911 - accuracy: 0.0984 - val_loss: 0.1143 - val_accuracy: 0.1021\n",
            "Epoch 42/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1935 - accuracy: 0.0986 - val_loss: 0.1525 - val_accuracy: 0.1019\n",
            "Epoch 43/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.2010 - accuracy: 0.0991 - val_loss: 0.1085 - val_accuracy: 0.1015\n",
            "Epoch 44/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1880 - accuracy: 0.0985 - val_loss: 0.1493 - val_accuracy: 0.0986\n",
            "Epoch 45/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1931 - accuracy: 0.0983 - val_loss: 0.2391 - val_accuracy: 0.0987\n",
            "Epoch 46/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1924 - accuracy: 0.0990 - val_loss: 0.1245 - val_accuracy: 0.1000\n",
            "Epoch 47/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1924 - accuracy: 0.0983 - val_loss: 0.1425 - val_accuracy: 0.1002\n",
            "Epoch 48/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1836 - accuracy: 0.0989 - val_loss: 0.1542 - val_accuracy: 0.1023\n",
            "Epoch 49/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1954 - accuracy: 0.0989 - val_loss: 0.1131 - val_accuracy: 0.0990\n",
            "Epoch 50/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1851 - accuracy: 0.0988 - val_loss: 0.1596 - val_accuracy: 0.1030\n",
            "Epoch 51/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1850 - accuracy: 0.0987 - val_loss: 0.1275 - val_accuracy: 0.1001\n",
            "Epoch 52/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1860 - accuracy: 0.0984 - val_loss: 0.1421 - val_accuracy: 0.1008\n",
            "Epoch 53/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1919 - accuracy: 0.0983 - val_loss: 0.1713 - val_accuracy: 0.0985\n",
            "Epoch 54/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1891 - accuracy: 0.0986 - val_loss: 0.1805 - val_accuracy: 0.1075\n",
            "Epoch 55/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1891 - accuracy: 0.0984 - val_loss: 0.1215 - val_accuracy: 0.1000\n",
            "Epoch 56/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1868 - accuracy: 0.0990 - val_loss: 0.1243 - val_accuracy: 0.0985\n",
            "Epoch 57/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1915 - accuracy: 0.0985 - val_loss: 0.1332 - val_accuracy: 0.0982\n",
            "Epoch 58/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1855 - accuracy: 0.0988 - val_loss: 0.1480 - val_accuracy: 0.1004\n",
            "Epoch 59/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1885 - accuracy: 0.0983 - val_loss: 0.2043 - val_accuracy: 0.1037\n",
            "Epoch 60/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1780 - accuracy: 0.0993 - val_loss: 0.1972 - val_accuracy: 0.1046\n",
            "Epoch 61/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1870 - accuracy: 0.0987 - val_loss: 0.1153 - val_accuracy: 0.0998\n",
            "Epoch 62/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1829 - accuracy: 0.0991 - val_loss: 0.1670 - val_accuracy: 0.0994\n",
            "Epoch 63/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1839 - accuracy: 0.0986 - val_loss: 0.1782 - val_accuracy: 0.1035\n",
            "Epoch 64/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1862 - accuracy: 0.0989 - val_loss: 0.1837 - val_accuracy: 0.0913\n",
            "Epoch 65/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1918 - accuracy: 0.0986 - val_loss: 0.1550 - val_accuracy: 0.0982\n",
            "Epoch 66/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1772 - accuracy: 0.0992 - val_loss: 0.1870 - val_accuracy: 0.1005\n",
            "Epoch 67/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1845 - accuracy: 0.0982 - val_loss: 0.1351 - val_accuracy: 0.0979\n",
            "Epoch 68/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1807 - accuracy: 0.0983 - val_loss: 0.1552 - val_accuracy: 0.0980\n",
            "Epoch 69/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1773 - accuracy: 0.0982 - val_loss: 0.1368 - val_accuracy: 0.1001\n",
            "Epoch 70/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1827 - accuracy: 0.0981 - val_loss: 0.1259 - val_accuracy: 0.0967\n",
            "Epoch 71/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1845 - accuracy: 0.0985 - val_loss: 0.1792 - val_accuracy: 0.1021\n",
            "Epoch 72/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1772 - accuracy: 0.0986 - val_loss: 0.1129 - val_accuracy: 0.1011\n",
            "Epoch 73/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1945 - accuracy: 0.0983 - val_loss: 0.1900 - val_accuracy: 0.1000\n",
            "Epoch 74/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1811 - accuracy: 0.0990 - val_loss: 0.1128 - val_accuracy: 0.1015\n",
            "Epoch 75/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1727 - accuracy: 0.0984 - val_loss: 0.1266 - val_accuracy: 0.0975\n",
            "Epoch 76/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1851 - accuracy: 0.0984 - val_loss: 0.1634 - val_accuracy: 0.0968\n",
            "Epoch 77/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1753 - accuracy: 0.0981 - val_loss: 0.1355 - val_accuracy: 0.1000\n",
            "Epoch 78/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1791 - accuracy: 0.0987 - val_loss: 0.1269 - val_accuracy: 0.1045\n",
            "Epoch 79/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1779 - accuracy: 0.0988 - val_loss: 0.1352 - val_accuracy: 0.1008\n",
            "Epoch 80/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1810 - accuracy: 0.0985 - val_loss: 0.1298 - val_accuracy: 0.1031\n",
            "Epoch 81/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1837 - accuracy: 0.0988 - val_loss: 0.1758 - val_accuracy: 0.1014\n",
            "Epoch 82/150\n",
            "788/788 [==============================] - 12s 15ms/step - loss: 0.1817 - accuracy: 0.0981 - val_loss: 0.2683 - val_accuracy: 0.1027\n",
            "Epoch 83/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1787 - accuracy: 0.0981 - val_loss: 0.1461 - val_accuracy: 0.1004\n",
            "Epoch 84/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1846 - accuracy: 0.0985 - val_loss: 0.1324 - val_accuracy: 0.1011\n",
            "Epoch 85/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1754 - accuracy: 0.0989 - val_loss: 0.1977 - val_accuracy: 0.1001\n",
            "Epoch 86/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1846 - accuracy: 0.0992 - val_loss: 0.1740 - val_accuracy: 0.1015\n",
            "Epoch 87/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1781 - accuracy: 0.0988 - val_loss: 0.1543 - val_accuracy: 0.1018\n",
            "Epoch 88/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1800 - accuracy: 0.0987 - val_loss: 0.1856 - val_accuracy: 0.1032\n",
            "Epoch 89/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1764 - accuracy: 0.0983 - val_loss: 0.1339 - val_accuracy: 0.0993\n",
            "Epoch 90/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1746 - accuracy: 0.0984 - val_loss: 0.1474 - val_accuracy: 0.1002\n",
            "Epoch 91/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1740 - accuracy: 0.0986 - val_loss: 0.1565 - val_accuracy: 0.0985\n",
            "Epoch 92/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1821 - accuracy: 0.0984 - val_loss: 0.2541 - val_accuracy: 0.1011\n",
            "Epoch 93/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1725 - accuracy: 0.0987 - val_loss: 0.1412 - val_accuracy: 0.0999\n",
            "Epoch 94/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1817 - accuracy: 0.0985 - val_loss: 0.1619 - val_accuracy: 0.1002\n",
            "Epoch 95/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1778 - accuracy: 0.0983 - val_loss: 0.1387 - val_accuracy: 0.1000\n",
            "Epoch 96/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1769 - accuracy: 0.0984 - val_loss: 0.3451 - val_accuracy: 0.1037\n",
            "Epoch 97/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1795 - accuracy: 0.0988 - val_loss: 0.3003 - val_accuracy: 0.1011\n",
            "Epoch 98/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1763 - accuracy: 0.0984 - val_loss: 0.1993 - val_accuracy: 0.0975\n",
            "Epoch 99/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1813 - accuracy: 0.0987 - val_loss: 0.1964 - val_accuracy: 0.0945\n",
            "Epoch 100/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1717 - accuracy: 0.0989 - val_loss: 0.1244 - val_accuracy: 0.0993\n",
            "Epoch 101/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1740 - accuracy: 0.0987 - val_loss: 0.6398 - val_accuracy: 0.0937\n",
            "Epoch 102/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1729 - accuracy: 0.0988 - val_loss: 0.6282 - val_accuracy: 0.1037\n",
            "Epoch 103/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1769 - accuracy: 0.0992 - val_loss: 0.1547 - val_accuracy: 0.1023\n",
            "Epoch 104/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1717 - accuracy: 0.0983 - val_loss: 0.1613 - val_accuracy: 0.1025\n",
            "Epoch 105/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1718 - accuracy: 0.0988 - val_loss: 0.1375 - val_accuracy: 0.1012\n",
            "Epoch 106/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1734 - accuracy: 0.0983 - val_loss: 0.3769 - val_accuracy: 0.0992\n",
            "Epoch 107/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1672 - accuracy: 0.0982 - val_loss: 0.1351 - val_accuracy: 0.0971\n",
            "Epoch 108/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1787 - accuracy: 0.0983 - val_loss: 0.1706 - val_accuracy: 0.0979\n",
            "Epoch 109/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1669 - accuracy: 0.0982 - val_loss: 0.2603 - val_accuracy: 0.1004\n",
            "Epoch 110/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1713 - accuracy: 0.0985 - val_loss: 0.6068 - val_accuracy: 0.1050\n",
            "Epoch 111/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1665 - accuracy: 0.0986 - val_loss: 9.5216 - val_accuracy: 0.0983\n",
            "Epoch 112/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1715 - accuracy: 0.0986 - val_loss: 0.1568 - val_accuracy: 0.1000\n",
            "Epoch 113/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1764 - accuracy: 0.0981 - val_loss: 0.1170 - val_accuracy: 0.1030\n",
            "Epoch 114/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1664 - accuracy: 0.0980 - val_loss: 0.1444 - val_accuracy: 0.1001\n",
            "Epoch 115/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1691 - accuracy: 0.0985 - val_loss: 0.1516 - val_accuracy: 0.1001\n",
            "Epoch 116/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1739 - accuracy: 0.0983 - val_loss: 0.2926 - val_accuracy: 0.1027\n",
            "Epoch 117/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1697 - accuracy: 0.0983 - val_loss: 1.5181 - val_accuracy: 0.1040\n",
            "Epoch 118/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1711 - accuracy: 0.0985 - val_loss: 0.1342 - val_accuracy: 0.0996\n",
            "Epoch 119/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1693 - accuracy: 0.0987 - val_loss: 0.2139 - val_accuracy: 0.1054\n",
            "Epoch 120/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1699 - accuracy: 0.0983 - val_loss: 0.4252 - val_accuracy: 0.1030\n",
            "Epoch 121/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1726 - accuracy: 0.0982 - val_loss: 0.4210 - val_accuracy: 0.0979\n",
            "Epoch 122/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1673 - accuracy: 0.0988 - val_loss: 0.1213 - val_accuracy: 0.1002\n",
            "Epoch 123/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1730 - accuracy: 0.0987 - val_loss: 0.6947 - val_accuracy: 0.1033\n",
            "Epoch 124/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1719 - accuracy: 0.0985 - val_loss: 0.9389 - val_accuracy: 0.1014\n",
            "Epoch 125/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1665 - accuracy: 0.0984 - val_loss: 0.1593 - val_accuracy: 0.1000\n",
            "Epoch 126/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1712 - accuracy: 0.0984 - val_loss: 0.1907 - val_accuracy: 0.1032\n",
            "Epoch 127/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1655 - accuracy: 0.0986 - val_loss: 0.1404 - val_accuracy: 0.1033\n",
            "Epoch 128/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1708 - accuracy: 0.0987 - val_loss: 0.8540 - val_accuracy: 0.1032\n",
            "Epoch 129/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1673 - accuracy: 0.0987 - val_loss: 0.2895 - val_accuracy: 0.1017\n",
            "Epoch 130/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1642 - accuracy: 0.0980 - val_loss: 0.1155 - val_accuracy: 0.0999\n",
            "Epoch 131/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1783 - accuracy: 0.0989 - val_loss: 0.1574 - val_accuracy: 0.0981\n",
            "Epoch 132/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1589 - accuracy: 0.0983 - val_loss: 0.2121 - val_accuracy: 0.1015\n",
            "Epoch 133/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1714 - accuracy: 0.0981 - val_loss: 0.1471 - val_accuracy: 0.1048\n",
            "Epoch 134/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1670 - accuracy: 0.0988 - val_loss: 0.1495 - val_accuracy: 0.1040\n",
            "Epoch 135/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1700 - accuracy: 0.0984 - val_loss: 0.1462 - val_accuracy: 0.0970\n",
            "Epoch 136/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1656 - accuracy: 0.0980 - val_loss: 0.1471 - val_accuracy: 0.1064\n",
            "Epoch 137/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1728 - accuracy: 0.0987 - val_loss: 0.1261 - val_accuracy: 0.0992\n",
            "Epoch 138/150\n",
            "788/788 [==============================] - 11s 15ms/step - loss: 0.1694 - accuracy: 0.0985 - val_loss: 0.1186 - val_accuracy: 0.1012\n",
            "Epoch 139/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1674 - accuracy: 0.0984 - val_loss: 0.1771 - val_accuracy: 0.1010\n",
            "Epoch 140/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1723 - accuracy: 0.0984 - val_loss: 0.1406 - val_accuracy: 0.1017\n",
            "Epoch 141/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1699 - accuracy: 0.0987 - val_loss: 0.1707 - val_accuracy: 0.1015\n",
            "Epoch 142/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1729 - accuracy: 0.0979 - val_loss: 0.1352 - val_accuracy: 0.1002\n",
            "Epoch 143/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1610 - accuracy: 0.0983 - val_loss: 0.1337 - val_accuracy: 0.0973\n",
            "Epoch 144/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1678 - accuracy: 0.0987 - val_loss: 0.1457 - val_accuracy: 0.1002\n",
            "Epoch 145/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1682 - accuracy: 0.0986 - val_loss: 0.1382 - val_accuracy: 0.1002\n",
            "Epoch 146/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1664 - accuracy: 0.0982 - val_loss: 0.1585 - val_accuracy: 0.1031\n",
            "Epoch 147/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1694 - accuracy: 0.0988 - val_loss: 0.3959 - val_accuracy: 0.0951\n",
            "Epoch 148/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1633 - accuracy: 0.0985 - val_loss: 1.0141 - val_accuracy: 0.0973\n",
            "Epoch 149/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1659 - accuracy: 0.0981 - val_loss: 0.6335 - val_accuracy: 0.0977\n",
            "Epoch 150/150\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.1606 - accuracy: 0.0985 - val_loss: 0.1434 - val_accuracy: 0.0989\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.src.callbacks.History at 0x79ec6827fe50>"
            ]
          },
          "metadata": {},
          "execution_count": 73
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "predictions_train=cnn_model.predict(x_train)\n",
        "predictions_train=np.argmax(predictions_train, axis=1)\n",
        "accuracy_train=accuracy_score(predictions_train, y_train)\n",
        "print(f\"accuracy score for train set is {accuracy_train}\")\n",
        "\n",
        "\n",
        "predictions_test=cnn_model.predict(x_test)\n",
        "predictions_test=np.argmax(predictions_test,axis=1)\n",
        "accuracy_test=accuracy_score(predictions_test, y_test)\n",
        "print(f\"accuracy score for test set is {accuracy_test}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "stZNUXMXVX2x",
        "outputId": "b618a0f9-7e9a-4aad-f2d0-ca8ef70be444"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "788/788 [==============================] - 3s 3ms/step\n",
            "accuracy score for train set is 0.9750396825396825\n",
            "263/263 [==============================] - 1s 4ms/step\n",
            "accuracy score for test set is 0.971547619047619\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "D_yxHRrv8Cc1"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "#LeNet-5"
      ],
      "metadata": {
        "id": "6EfRk76E8DnQ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#sequential api\n",
        "cnn_model3=Sequential([\n",
        "    Conv2D(filters=6,kernel_size=5,strides=1),\n",
        "    AveragePooling2D(pool_size=(2,2), strides=2),\n",
        "    Conv2D(filters=16,kernel_size=5, strides=1),\n",
        "    AveragePooling2D(pool_size=2,strides=2),\n",
        "    Flatten(),\n",
        "    Dense(120,'relu'),\n",
        "    Dense(10,'softmax')\n",
        "])\n",
        "\n",
        "cnn_model3.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),\n",
        "                   optimizer=tf.keras.optimizers.Adam(0.001),\n",
        "                   metrics=['accuracy'])"
      ],
      "metadata": {
        "id": "kEUq12g48GGr"
      },
      "execution_count": 22,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn_model3.fit(x_traincv, y_traincv, epochs=20)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tWpeEhkQ-fUl",
        "outputId": "50cfbbe2-20cd-4b59-9076-3767e5b91e06"
      },
      "execution_count": 67,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/20\n",
            "1050/1050 [==============================] - 17s 15ms/step - loss: 0.2538 - accuracy: 0.9243\n",
            "Epoch 2/20\n",
            "1050/1050 [==============================] - 15s 15ms/step - loss: 0.1209 - accuracy: 0.9630\n",
            "Epoch 3/20\n",
            "1050/1050 [==============================] - 25s 24ms/step - loss: 0.0939 - accuracy: 0.9704\n",
            "Epoch 4/20\n",
            "1050/1050 [==============================] - 18s 17ms/step - loss: 0.0775 - accuracy: 0.9748\n",
            "Epoch 5/20\n",
            "1050/1050 [==============================] - 24s 23ms/step - loss: 0.0653 - accuracy: 0.9788\n",
            "Epoch 6/20\n",
            "1050/1050 [==============================] - 27s 26ms/step - loss: 0.0574 - accuracy: 0.9810\n",
            "Epoch 7/20\n",
            "1050/1050 [==============================] - 26s 24ms/step - loss: 0.0518 - accuracy: 0.9829\n",
            "Epoch 8/20\n",
            "1050/1050 [==============================] - 24s 23ms/step - loss: 0.0453 - accuracy: 0.9853\n",
            "Epoch 9/20\n",
            "1050/1050 [==============================] - 18s 17ms/step - loss: 0.0409 - accuracy: 0.9865\n",
            "Epoch 10/20\n",
            "1050/1050 [==============================] - 20s 19ms/step - loss: 0.0368 - accuracy: 0.9876\n",
            "Epoch 11/20\n",
            "1050/1050 [==============================] - 26s 25ms/step - loss: 0.0336 - accuracy: 0.9883\n",
            "Epoch 12/20\n",
            "1050/1050 [==============================] - 22s 21ms/step - loss: 0.0331 - accuracy: 0.9892\n",
            "Epoch 13/20\n",
            "1050/1050 [==============================] - 25s 24ms/step - loss: 0.0291 - accuracy: 0.9892\n",
            "Epoch 14/20\n",
            "1050/1050 [==============================] - 16s 16ms/step - loss: 0.0242 - accuracy: 0.9917\n",
            "Epoch 15/20\n",
            "1050/1050 [==============================] - 16s 15ms/step - loss: 0.0262 - accuracy: 0.9909\n",
            "Epoch 16/20\n",
            "1050/1050 [==============================] - 16s 15ms/step - loss: 0.0233 - accuracy: 0.9916\n",
            "Epoch 17/20\n",
            "1050/1050 [==============================] - 16s 15ms/step - loss: 0.0199 - accuracy: 0.9930\n",
            "Epoch 18/20\n",
            "1050/1050 [==============================] - 15s 15ms/step - loss: 0.0217 - accuracy: 0.9925\n",
            "Epoch 19/20\n",
            "1050/1050 [==============================] - 16s 15ms/step - loss: 0.0159 - accuracy: 0.9947\n",
            "Epoch 20/20\n",
            "1050/1050 [==============================] - 16s 15ms/step - loss: 0.0225 - accuracy: 0.9925\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.src.callbacks.History at 0x7af8540c3e80>"
            ]
          },
          "metadata": {},
          "execution_count": 67
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cnn_model3.fit(x_train,y_train, epochs=24,initial_epoch=21)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9VcUADTRPqE_",
        "outputId": "49c13687-3040-44cb-9582-acc09d9240e0"
      },
      "execution_count": 34,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 22/24\n",
            "788/788 [==============================] - 13s 16ms/step - loss: 0.0429 - accuracy: 0.9906\n",
            "Epoch 23/24\n",
            "788/788 [==============================] - 12s 15ms/step - loss: 0.0161 - accuracy: 0.9952\n",
            "Epoch 24/24\n",
            "788/788 [==============================] - 11s 14ms/step - loss: 0.0158 - accuracy: 0.9947\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.src.callbacks.History at 0x7e949d5c06d0>"
            ]
          },
          "metadata": {},
          "execution_count": 34
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cnn_model3.save_weights('lenet.h5')"
      ],
      "metadata": {
        "id": "jjFLuLkHI9Zz"
      },
      "execution_count": 35,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn_model3.load_weights('lenet.h5')"
      ],
      "metadata": {
        "id": "pAJVD94yV-C-"
      },
      "execution_count": 24,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_predictions=cnn_model3.predict(x_train)\n",
        "train_predictions=np.argmax(train_predictions,axis=1)\n",
        "train_accuracy=accuracy_score(train_predictions, y_train)\n",
        "\n",
        "test_predictions=cnn_model3.predict(x_test)\n",
        "test_predictions=np.argmax(test_predictions,axis=1)\n",
        "test_accuracy=accuracy_score(test_predictions, y_test)\n",
        "\n",
        "print(\"LeNet-5 architecture performance:\")\n",
        "print(f\"training set accuracy: {train_accuracy}\\ntest set accuracy: {test_accuracy}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "z8OeKDSk-fE2",
        "outputId": "40ee0926-e692-498b-c4d7-57fb9cbd7c41"
      },
      "execution_count": 36,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "788/788 [==============================] - 5s 6ms/step\n",
            "263/263 [==============================] - 2s 6ms/step\n",
            "LeNet-5 architecture performance:\n",
            "training set accuracy: 0.9959126984126984\n",
            "test set accuracy: 0.9873809523809524\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#functional api\n",
        "input=tf.keras.Input(shape=(28,28,1))\n",
        "x=Conv2D(filters=6,kernel_size=5,strides=1)(input)\n",
        "x=AveragePooling2D(pool_size=(2,2),strides=2)(x)\n",
        "x=Conv2D(filters=16,kernel_size=5, strides=1)(x)\n",
        "x=AveragePooling2D(pool_size=2,strides=2)(x)\n",
        "x=Flatten()(x)\n",
        "x=Dense(120,activation='relu')(x)\n",
        "x=Dense(84,activation='relu')(x)\n",
        "output=Dense(10,activation='softmax')(x)\n",
        "\n",
        "cnn_model4=tf.keras.Model(inputs=input,outputs=output)"
      ],
      "metadata": {
        "id": "5e8mZp0DDSzT"
      },
      "execution_count": 37,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cnn_model4.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),\n",
        "                   optimizer=tf.keras.optimizers.Adam(0.001),\n",
        "                   metrics=['accuracy'])\n",
        "cnn_model4.fit(x_traincv, y_traincv, epochs=10)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RiioVXzrDanT",
        "outputId": "51f6868d-cd65-43ac-91b7-e74fc786de92"
      },
      "execution_count": 39,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/10\n",
            "1050/1050 [==============================] - 19s 15ms/step - loss: 0.1602 - accuracy: 0.9504\n",
            "Epoch 2/10\n",
            "1050/1050 [==============================] - 16s 15ms/step - loss: 0.1049 - accuracy: 0.9661\n",
            "Epoch 3/10\n",
            "1050/1050 [==============================] - 16s 15ms/step - loss: 0.0815 - accuracy: 0.9743\n",
            "Epoch 4/10\n",
            "1050/1050 [==============================] - 16s 15ms/step - loss: 0.0671 - accuracy: 0.9777\n",
            "Epoch 5/10\n",
            "1050/1050 [==============================] - 16s 15ms/step - loss: 0.0598 - accuracy: 0.9792\n",
            "Epoch 6/10\n",
            "1050/1050 [==============================] - 16s 15ms/step - loss: 0.0505 - accuracy: 0.9829\n",
            "Epoch 7/10\n",
            "1050/1050 [==============================] - 16s 15ms/step - loss: 0.0430 - accuracy: 0.9859\n",
            "Epoch 8/10\n",
            "1050/1050 [==============================] - 16s 15ms/step - loss: 0.0393 - accuracy: 0.9867\n",
            "Epoch 9/10\n",
            "1050/1050 [==============================] - 16s 15ms/step - loss: 0.0354 - accuracy: 0.9889\n",
            "Epoch 10/10\n",
            "1050/1050 [==============================] - 16s 15ms/step - loss: 0.0328 - accuracy: 0.9887\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.src.callbacks.History at 0x7e9495093250>"
            ]
          },
          "metadata": {},
          "execution_count": 39
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cnn_model4.summary()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_xIgfLKEYOZ_",
        "outputId": "98cc938b-ef27-443c-d264-438617638caa"
      },
      "execution_count": 40,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"model\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " input_1 (InputLayer)        [(None, 28, 28, 1)]       0         \n",
            "                                                                 \n",
            " conv2d_4 (Conv2D)           (None, 24, 24, 6)         156       \n",
            "                                                                 \n",
            " average_pooling2d_2 (Avera  (None, 12, 12, 6)         0         \n",
            " gePooling2D)                                                    \n",
            "                                                                 \n",
            " conv2d_5 (Conv2D)           (None, 8, 8, 16)          2416      \n",
            "                                                                 \n",
            " average_pooling2d_3 (Avera  (None, 4, 4, 16)          0         \n",
            " gePooling2D)                                                    \n",
            "                                                                 \n",
            " flatten_2 (Flatten)         (None, 256)               0         \n",
            "                                                                 \n",
            " dense_4 (Dense)             (None, 120)               30840     \n",
            "                                                                 \n",
            " dense_5 (Dense)             (None, 84)                10164     \n",
            "                                                                 \n",
            " dense_6 (Dense)             (None, 10)                850       \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 44426 (173.54 KB)\n",
            "Trainable params: 44426 (173.54 KB)\n",
            "Non-trainable params: 0 (0.00 Byte)\n",
            "_________________________________________________________________\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#AlexNet"
      ],
      "metadata": {
        "id": "xmCXfy_pYO3q"
      }
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "YUUfoL7GYRWm"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "#test predictions"
      ],
      "metadata": {
        "id": "2mof638Tehve"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "print(predictions_train.shape, x_train.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_4Ug_EtfWK3A",
        "outputId": "b4fa7f5a-bf80-482b-c097-1efd68f2a4b0"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(25200,) (25200, 28, 28, 1)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "predictions_sub=cnn_model3.predict(d_test)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hZEzaeywX6l2",
        "outputId": "a1f35bf6-9582-4f00-a652-9a5e3879dc4f"
      },
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "875/875 [==============================] - 5s 6ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "predictions_sub=np.argmax(predictions_sub, axis=1)\n",
        "print(predictions_sub.shape, d_test.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_YKpytzQYOti",
        "outputId": "c898b2a6-29df-4b89-b8c5-744b6cbd259b"
      },
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(28000,) (28000, 28, 28, 1)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "ans=[]\n",
        "for i in range(len(predictions_sub)):\n",
        "  ans.append([i+1,predictions_sub[i]])"
      ],
      "metadata": {
        "id": "QQrnsBh1YVZC"
      },
      "execution_count": 29,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sub=pd.DataFrame(ans, columns=['ImageId','Label'])"
      ],
      "metadata": {
        "id": "Ibj36CgYZOO-"
      },
      "execution_count": 30,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sub.to_csv('submission.csv', index=False)"
      ],
      "metadata": {
        "id": "4jYsLxM-ZQtl"
      },
      "execution_count": 31,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "s=pd.read_csv('submission.csv')"
      ],
      "metadata": {
        "id": "PwzPr721ZpbV"
      },
      "execution_count": 32,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "s.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "CHOymU-9Z2FG",
        "outputId": "fba3cea7-dce7-4232-d479-e05256fd8a39"
      },
      "execution_count": 33,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   ImageId  Label\n",
              "0        1      2\n",
              "1        2      0\n",
              "2        3      9\n",
              "3        4      0\n",
              "4        5      3"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-38d47cf1-cfa4-415d-878d-9686882c3734\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>ImageId</th>\n",
              "      <th>Label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3</td>\n",
              "      <td>9</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>5</td>\n",
              "      <td>3</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-38d47cf1-cfa4-415d-878d-9686882c3734')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-38d47cf1-cfa4-415d-878d-9686882c3734 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-38d47cf1-cfa4-415d-878d-9686882c3734');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-da63f54d-b018-42c0-9113-ce4d62837019\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-da63f54d-b018-42c0-9113-ce4d62837019')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-da63f54d-b018-42c0-9113-ce4d62837019 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "s",
              "summary": "{\n  \"name\": \"s\",\n  \"rows\": 28000,\n  \"fields\": [\n    {\n      \"column\": \"ImageId\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 8083,\n        \"min\": 1,\n        \"max\": 28000,\n        \"num_unique_values\": 28000,\n        \"samples\": [\n          18407,\n          5035,\n          18326\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Label\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 2,\n        \"min\": 0,\n        \"max\": 9,\n        \"num_unique_values\": 10,\n        \"samples\": [\n          6,\n          0,\n          5\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 33
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "Mvbd7j5bZ7Dm"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "N7bqX3yiwo6U"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}