{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "private_outputs": true,
      "provenance": [],
      "gpuType": "V100"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "## Download & Import"
      ],
      "metadata": {
        "id": "QcyCfMBWBriR"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "-FM1OCyEBmgT"
      },
      "outputs": [],
      "source": [
        "!pip install transformers"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!unzip /content/drive/MyDrive/VQA/open.zip"
      ],
      "metadata": {
        "id": "BSP911cfBuQa"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "import pandas as pd\n",
        "\n",
        "import torch\n",
        "import torch.optim as optim\n",
        "import torch.nn as nn\n",
        "from torch.utils.data import Dataset, DataLoader\n",
        "\n",
        "from torchvision import transforms\n",
        "from PIL import Image\n",
        "\n",
        "from transformers import ViltProcessor, ViltForQuestionAnswering\n",
        "\n",
        "from tqdm.auto import tqdm"
      ],
      "metadata": {
        "id": "CWtNgpF0BvWu"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Answer Preprocessing & Labeling"
      ],
      "metadata": {
        "id": "RDB_DDoBBzzz"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import re\n",
        "\n",
        "contractions = {\n",
        "    \"aint\": \"ain't\",\n",
        "    \"arent\": \"aren't\",\n",
        "    \"cant\": \"can't\",\n",
        "    \"couldve\": \"could've\",\n",
        "    \"couldnt\": \"couldn't\",\n",
        "    \"couldn'tve\": \"couldn't've\",\n",
        "    \"couldnt've\": \"couldn't've\",\n",
        "    \"didnt\": \"didn't\",\n",
        "    \"doesnt\": \"doesn't\",\n",
        "    \"dont\": \"don't\",\n",
        "    \"hadnt\": \"hadn't\",\n",
        "    \"hadnt've\": \"hadn't've\",\n",
        "    \"hadn'tve\": \"hadn't've\",\n",
        "    \"hasnt\": \"hasn't\",\n",
        "    \"havent\": \"haven't\",\n",
        "    \"hed\": \"he'd\",\n",
        "    \"hed've\": \"he'd've\",\n",
        "    \"he'dve\": \"he'd've\",\n",
        "    \"hes\": \"he's\",\n",
        "    \"howd\": \"how'd\",\n",
        "    \"howll\": \"how'll\",\n",
        "    \"hows\": \"how's\",\n",
        "    \"Id've\": \"I'd've\",\n",
        "    \"I'dve\": \"I'd've\",\n",
        "    \"Im\": \"I'm\",\n",
        "    \"Ive\": \"I've\",\n",
        "    \"isnt\": \"isn't\",\n",
        "    \"itd\": \"it'd\",\n",
        "    \"itd've\": \"it'd've\",\n",
        "    \"it'dve\": \"it'd've\",\n",
        "    \"itll\": \"it'll\",\n",
        "    \"let's\": \"let's\",\n",
        "    \"maam\": \"ma'am\",\n",
        "    \"mightnt\": \"mightn't\",\n",
        "    \"mightnt've\": \"mightn't've\",\n",
        "    \"mightn'tve\": \"mightn't've\",\n",
        "    \"mightve\": \"might've\",\n",
        "    \"mustnt\": \"mustn't\",\n",
        "    \"mustve\": \"must've\",\n",
        "    \"neednt\": \"needn't\",\n",
        "    \"notve\": \"not've\",\n",
        "    \"oclock\": \"o'clock\",\n",
        "    \"oughtnt\": \"oughtn't\",\n",
        "    \"ow's'at\": \"'ow's'at\",\n",
        "    \"'ows'at\": \"'ow's'at\",\n",
        "    \"'ow'sat\": \"'ow's'at\",\n",
        "    \"shant\": \"shan't\",\n",
        "    \"shed've\": \"she'd've\",\n",
        "    \"she'dve\": \"she'd've\",\n",
        "    \"she's\": \"she's\",\n",
        "    \"shouldve\": \"should've\",\n",
        "    \"shouldnt\": \"shouldn't\",\n",
        "    \"shouldnt've\": \"shouldn't've\",\n",
        "    \"shouldn'tve\": \"shouldn't've\",\n",
        "    \"somebody'd\": \"somebodyd\",\n",
        "    \"somebodyd've\": \"somebody'd've\",\n",
        "    \"somebody'dve\": \"somebody'd've\",\n",
        "    \"somebodyll\": \"somebody'll\",\n",
        "    \"somebodys\": \"somebody's\",\n",
        "    \"someoned\": \"someone'd\",\n",
        "    \"someoned've\": \"someone'd've\",\n",
        "    \"someone'dve\": \"someone'd've\",\n",
        "    \"someonell\": \"someone'll\",\n",
        "    \"someones\": \"someone's\",\n",
        "    \"somethingd\": \"something'd\",\n",
        "    \"somethingd've\": \"something'd've\",\n",
        "    \"something'dve\": \"something'd've\",\n",
        "    \"somethingll\": \"something'll\",\n",
        "    \"thats\": \"that's\",\n",
        "    \"thered\": \"there'd\",\n",
        "    \"thered've\": \"there'd've\",\n",
        "    \"there'dve\": \"there'd've\",\n",
        "    \"therere\": \"there're\",\n",
        "    \"theres\": \"there's\",\n",
        "    \"theyd\": \"they'd\",\n",
        "    \"theyd've\": \"they'd've\",\n",
        "    \"they'dve\": \"they'd've\",\n",
        "    \"theyll\": \"they'll\",\n",
        "    \"theyre\": \"they're\",\n",
        "    \"theyve\": \"they've\",\n",
        "    \"twas\": \"'twas\",\n",
        "    \"wasnt\": \"wasn't\",\n",
        "    \"wed've\": \"we'd've\",\n",
        "    \"we'dve\": \"we'd've\",\n",
        "    \"weve\": \"we've\",\n",
        "    \"werent\": \"weren't\",\n",
        "    \"whatll\": \"what'll\",\n",
        "    \"whatre\": \"what're\",\n",
        "    \"whats\": \"what's\",\n",
        "    \"whatve\": \"what've\",\n",
        "    \"whens\": \"when's\",\n",
        "    \"whered\": \"where'd\",\n",
        "    \"wheres\": \"where's\",\n",
        "    \"whereve\": \"where've\",\n",
        "    \"whod\": \"who'd\",\n",
        "    \"whod've\": \"who'd've\",\n",
        "    \"who'dve\": \"who'd've\",\n",
        "    \"wholl\": \"who'll\",\n",
        "    \"whos\": \"who's\",\n",
        "    \"whove\": \"who've\",\n",
        "    \"whyll\": \"why'll\",\n",
        "    \"whyre\": \"why're\",\n",
        "    \"whys\": \"why's\",\n",
        "    \"wont\": \"won't\",\n",
        "    \"wouldve\": \"would've\",\n",
        "    \"wouldnt\": \"wouldn't\",\n",
        "    \"wouldnt've\": \"wouldn't've\",\n",
        "    \"wouldn'tve\": \"wouldn't've\",\n",
        "    \"yall\": \"y'all\",\n",
        "    \"yall'll\": \"y'all'll\",\n",
        "    \"y'allll\": \"y'all'll\",\n",
        "    \"yall'd've\": \"y'all'd've\",\n",
        "    \"y'alld've\": \"y'all'd've\",\n",
        "    \"y'all'dve\": \"y'all'd've\",\n",
        "    \"youd\": \"you'd\",\n",
        "    \"youd've\": \"you'd've\",\n",
        "    \"you'dve\": \"you'd've\",\n",
        "    \"youll\": \"you'll\",\n",
        "    \"youre\": \"you're\",\n",
        "    \"youve\": \"you've\",\n",
        "}\n",
        "\n",
        "manual_map = {\n",
        "    \"none\": \"0\",\n",
        "    \"zero\": \"0\",\n",
        "    \"one\": \"1\",\n",
        "    \"two\": \"2\",\n",
        "    \"three\": \"3\",\n",
        "    \"four\": \"4\",\n",
        "    \"five\": \"5\",\n",
        "    \"six\": \"6\",\n",
        "    \"seven\": \"7\",\n",
        "    \"eight\": \"8\",\n",
        "    \"nine\": \"9\",\n",
        "    \"ten\": \"10\",\n",
        "}\n",
        "articles = [\"a\", \"an\", \"the\"]\n",
        "period_strip = re.compile(\"(?!<=\\d)(\\.)(?!\\d)\")\n",
        "comma_strip = re.compile(\"(\\d)(\\,)(\\d)\")\n",
        "punct = [\n",
        "    \";\",\n",
        "    r\"/\",\n",
        "    \"[\",\n",
        "    \"]\",\n",
        "    '\"',\n",
        "    \"{\",\n",
        "    \"}\",\n",
        "    \"(\",\n",
        "    \")\",\n",
        "    \"=\",\n",
        "    \"+\",\n",
        "    \"\\\\\",\n",
        "    \"_\",\n",
        "    \"-\",\n",
        "    \">\",\n",
        "    \"<\",\n",
        "    \"@\",\n",
        "    \"`\",\n",
        "    \",\",\n",
        "    \"?\",\n",
        "    \"!\",\n",
        "]\n",
        "\n",
        "\n",
        "def normalize_word(token):\n",
        "    _token = token\n",
        "    for p in punct:\n",
        "        if (p + \" \" in token or \" \" + p in token) or (\n",
        "            re.search(comma_strip, token) != None\n",
        "        ):\n",
        "            _token = _token.replace(p, \"\")\n",
        "        else:\n",
        "            _token = _token.replace(p, \" \")\n",
        "    token = period_strip.sub(\"\", _token, re.UNICODE)\n",
        "\n",
        "    _token = []\n",
        "    temp = token.lower().split()\n",
        "    for word in temp:\n",
        "        word = manual_map.setdefault(word, word)\n",
        "        if word not in articles:\n",
        "            _token.append(word)\n",
        "    for i, word in enumerate(_token):\n",
        "        if word in contractions:\n",
        "            _token[i] = contractions[word]\n",
        "    token = \" \".join(_token)\n",
        "    token = token.replace(\",\", \"\")\n",
        "    return token"
      ],
      "metadata": {
        "id": "wI7itV2qB0ZT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "\n",
        "df = pd.read_csv('train.csv')\n",
        "\n",
        "from collections import Counter\n",
        "\n",
        "# The original function to assign scores based on the occurences\n",
        "def get_score(occurences):\n",
        "    if occurences == 0:\n",
        "        return 0.0\n",
        "    elif occurences < 1:\n",
        "        return 0.3\n",
        "    elif occurences < 3:\n",
        "        return 0.6\n",
        "    elif occurences < 5:\n",
        "        return 0.9\n",
        "    else:\n",
        "        return 1.0\n",
        "\n",
        "# Apply the normalize_word function to the 'answer' column\n",
        "df['answer'] = df['answer'].apply(normalize_word)\n",
        "\n",
        "# Count the occurences of each answer\n",
        "answer_counter = Counter(df['answer'])\n",
        "\n",
        "# Filter answers with frequency 9 or more\n",
        "frequent_answers = [answer for answer, freq in answer_counter.items() if freq >= 5]\n",
        "\n",
        "# Filter the dataframe to keep only the rows with the frequent answers\n",
        "df = df[df['answer'].isin(frequent_answers)]\n",
        "\n",
        "# Create a new column for the occurences\n",
        "df['answer_occurences'] = df['answer'].apply(lambda x: answer_counter[x])\n",
        "\n",
        "# Create a new column for the scores\n",
        "df['answer_scores'] = df['answer_occurences'].apply(get_score)\n",
        "\n",
        "# Create a new column for the labels\n",
        "# The labels are created by enumerating the unique answers\n",
        "unique_answers = df['answer'].unique()\n",
        "answer2label = {k: i for i, k in enumerate(unique_answers)}\n",
        "\n",
        "answer_counter = Counter(df['answer'])\n",
        "label2answer = list(answer_counter.keys())\n",
        "\n",
        "df['answer_labels'] = df['answer'].apply(lambda x: answer2label[x])"
      ],
      "metadata": {
        "id": "H1nTFzYbCZeQ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df.to_csv('train_updated.csv', index=False)"
      ],
      "metadata": {
        "id": "WVb7rNqVD2Hf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "len(answer2label)"
      ],
      "metadata": {
        "id": "5mZdZcL9FuCj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "len(label2answer)"
      ],
      "metadata": {
        "id": "kQvQ_w9LGGnI"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import pickle\n",
        "\n",
        "# Save the answer2label and label2answer mappings\n",
        "with open('answer2label.pkl', 'wb') as f:\n",
        "    pickle.dump(answer2label, f)\n",
        "\n",
        "with open('label2answer.pkl', 'wb') as f:\n",
        "    pickle.dump(label2answer, f)"
      ],
      "metadata": {
        "id": "k0RLe17OHVFs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import pickle\n",
        "\n",
        "with open('answer2label.pkl', 'rb') as f:\n",
        "    answer2label = pickle.load(f)\n",
        "\n",
        "with open('label2answer.pkl', 'rb') as f:\n",
        "    label2answer = pickle.load(f)"
      ],
      "metadata": {
        "id": "J9euNgS9HWaZ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Augmentation"
      ],
      "metadata": {
        "id": "4NNtbWkDIfBl"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import random\n",
        "\n",
        "import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw\n",
        "import numpy as np\n",
        "import torch\n",
        "from PIL import Image"
      ],
      "metadata": {
        "id": "MOSXdhyNJKuc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def ShearX(img, v):  # [-0.3, 0.3]\n",
        "    assert -0.3 <= v <= 0.3\n",
        "    if random.random() > 0.5:\n",
        "        v = -v\n",
        "    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))\n",
        "\n",
        "def ShearY(img, v):  # [-0.3, 0.3]\n",
        "    assert -0.3 <= v <= 0.3\n",
        "    if random.random() > 0.5:\n",
        "        v = -v\n",
        "    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))\n",
        "\n",
        "def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]\n",
        "    assert -0.45 <= v <= 0.45\n",
        "    if random.random() > 0.5:\n",
        "        v = -v\n",
        "    v = v * img.size[0]\n",
        "    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))\n",
        "\n",
        "def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]\n",
        "    assert 0 <= v\n",
        "    if random.random() > 0.5:\n",
        "        v = -v\n",
        "    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))\n",
        "\n",
        "def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]\n",
        "    assert -0.45 <= v <= 0.45\n",
        "    if random.random() > 0.5:\n",
        "        v = -v\n",
        "    v = v * img.size[1]\n",
        "    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))\n",
        "\n",
        "def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]\n",
        "    assert 0 <= v\n",
        "    if random.random() > 0.5:\n",
        "        v = -v\n",
        "    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))\n",
        "\n",
        "def Rotate(img, v):  # [-30, 30]\n",
        "    assert -30 <= v <= 30\n",
        "    if random.random() > 0.5:\n",
        "        v = -v\n",
        "    return img.rotate(v)\n",
        "\n",
        "def Flip(img, _):  # not from the paper\n",
        "    return PIL.ImageOps.mirror(img)\n",
        "\n",
        "def Identity(img, v):\n",
        "    return img\n",
        "\n",
        "def augment_list():\n",
        "    l = [\n",
        "        (Identity, 0., 1.0),\n",
        "        (ShearX, 0., 0.3),  # 0\n",
        "        (ShearY, 0., 0.3),  # 1\n",
        "        (TranslateX, 0., 0.33),  # 2\n",
        "        (TranslateY, 0., 0.33),  # 3\n",
        "        (Rotate, 0, 30),  # 4\n",
        "        (Flip, 0, 1),  # 5\n",
        "        (TranslateXabs, 0.0, 100),\n",
        "        (TranslateYabs, 0.0, 100),\n",
        "    ]\n",
        "\n",
        "    return l\n",
        "\n",
        "class RandAugment:\n",
        "    def __init__(self, n, m):\n",
        "        self.n = n\n",
        "        self.m = m  # [0, 30]\n",
        "        self.augment_list = augment_list()\n",
        "\n",
        "    def __call__(self, img):\n",
        "        ops = random.choices(self.augment_list, k=self.n)\n",
        "        for op, minval, maxval in ops:\n",
        "            val = (float(self.m) / 30) * float(maxval - minval) + minval\n",
        "            img = op(img, val)\n",
        "\n",
        "        return img\n",
        "\n"
      ],
      "metadata": {
        "id": "iuH7QCmWIgjJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Dataset"
      ],
      "metadata": {
        "id": "RdmnS69DIziK"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "class VQADataset(Dataset):\n",
        "    def __init__(self, df, processor, img_path, is_test=False):\n",
        "        self.df = df\n",
        "        self.processor = processor\n",
        "        self.img_path = img_path\n",
        "        self.is_test = is_test\n",
        "\n",
        "        self.randaugment = RandAugment(2, 9)  # You can adjust the parameters as needed\n",
        "\n",
        "    def __len__(self):\n",
        "        return len(self.df)\n",
        "\n",
        "    def __getitem__(self, idx):\n",
        "        row = self.df.iloc[idx]\n",
        "\n",
        "        img_name = os.path.join(self.img_path, row['image_id'] + '.jpg')\n",
        "        image = PIL.Image.open(img_name).convert('RGB')\n",
        "\n",
        "        # Apply RandAugment if not in test mode\n",
        "        if not self.is_test:\n",
        "            image = self.randaugment(image)\n",
        "\n",
        "        question = row['question']\n",
        "        question = question.replace('?', ' ? ')\n",
        "        question = question.replace('.', ' . ')\n",
        "        question = question.replace(',', ' . ')\n",
        "        question = question.replace('!', ' . ')\n",
        "\n",
        "        max_length = 40\n",
        "        encoding = self.processor(image, question, padding=\"max_length\", return_tensors=\"pt\")\n",
        "        for k,v in encoding.items():\n",
        "          encoding[k] = v.squeeze()\n",
        "\n",
        "        if not self.is_test:\n",
        "            answer = row['answer']\n",
        "            label = row['answer_labels']\n",
        "            score = row['answer_scores']\n",
        "            targets = torch.zeros(len(answer2label))\n",
        "            targets[label] = score\n",
        "            encoding['labels'] = targets\n",
        "\n",
        "        else :\n",
        "            targets = torch.zeros(len(answer2label))\n",
        "            encoding['labels'] = targets\n",
        "\n",
        "        return encoding\n"
      ],
      "metadata": {
        "id": "RtSTjqoNI03T"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def collate_fn(batch):\n",
        "  input_ids = [item['input_ids'] for item in batch]\n",
        "  pixel_values = [item['pixel_values'] for item in batch]\n",
        "  attention_mask = [item['attention_mask'] for item in batch]\n",
        "  token_type_ids = [item['token_type_ids'] for item in batch]\n",
        "  labels = [item['labels'] for item in batch]\n",
        "\n",
        "  # create padded pixel values and corresponding pixel mask\n",
        "  encoding = processor.image_processor.pad(pixel_values, return_tensors=\"pt\")\n",
        "\n",
        "  # create new batch\n",
        "  batch = {}\n",
        "  batch['input_ids'] = torch.stack(input_ids)\n",
        "  batch['attention_mask'] = torch.stack(attention_mask)\n",
        "  batch['token_type_ids'] = torch.stack(token_type_ids)\n",
        "  batch['pixel_values'] = encoding['pixel_values']\n",
        "  batch['pixel_mask'] = encoding['pixel_mask']\n",
        "  batch['labels'] = torch.stack(labels)\n",
        "\n",
        "  return batch"
      ],
      "metadata": {
        "id": "wJ4-CEmzLX9Z"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_df = pd.read_csv('train_updated.csv')\n",
        "test_df = pd.read_csv('test.csv')\n",
        "sample_submission = pd.read_csv('sample_submission.csv')\n",
        "train_img_path = 'image/train'\n",
        "test_img_path = 'image/test'"
      ],
      "metadata": {
        "id": "y_m9ra-wLZbH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "processor = ViltProcessor.from_pretrained(\"dandelin/vilt-b32-mlm\")\n",
        "\n",
        "train_dataset = VQADataset(train_df, processor, train_img_path, is_test=False)\n",
        "train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=32, shuffle=True)"
      ],
      "metadata": {
        "id": "H6nj2_wPLc16"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Train"
      ],
      "metadata": {
        "id": "kBPc9VxrLz44"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def train(model, train_loader, optimizer, scheduler):\n",
        "    model.train()\n",
        "    total_loss = 0\n",
        "\n",
        "    # Additional variable to keep track of the number of steps\n",
        "    step = 0\n",
        "    total_steps = len(train_loader)\n",
        "\n",
        "    for batch in tqdm(train_loader):\n",
        "        batch = {k: v.to(device) for k, v in batch.items()}\n",
        "        optimizer.zero_grad()\n",
        "        outputs = model(**batch)\n",
        "        loss = outputs.loss\n",
        "\n",
        "        total_loss += loss.item()\n",
        "\n",
        "        # Only print the loss for every 100 steps\n",
        "        if step % 100 == 0:\n",
        "            print(\"Step:\", step, \"Loss:\", loss.item())\n",
        "\n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "\n",
        "        # Update the learning rate using the scheduler\n",
        "        scheduler.step(epoch=(step / total_steps))\n",
        "\n",
        "        # Increment the step count\n",
        "        step += 1\n",
        "\n",
        "    avg_loss = total_loss / len(train_loader)\n",
        "    return avg_loss\n",
        "\n",
        "def inference(model, loader):\n",
        "    model.eval()\n",
        "    preds = []\n",
        "    with torch.no_grad():\n",
        "        for data in tqdm(loader, total=len(loader)):\n",
        "            data = {k: v.to(device) for k, v in data.items()}\n",
        "            outputs = model(**data)\n",
        "\n",
        "            logits = outputs.logits\n",
        "            predicted_class = logits.argmax(-1)\n",
        "            pred = [label2answer[pred_class.item()] for pred_class in predicted_class]\n",
        "\n",
        "            preds.extend(pred)\n",
        "\n",
        "    return preds\n",
        "\n"
      ],
      "metadata": {
        "id": "LEpUDftlL09u"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')\n",
        "print(f\"current device is {device}\")\n",
        "\n",
        "model = ViltForQuestionAnswering.from_pretrained(\"dandelin/vilt-b32-mlm-itm\",\n",
        "                                                 hidden_dropout_prob = 0.1,\n",
        "                                                 attention_probs_dropout_prob = 0.1,\n",
        "                                                 num_labels=len(answer2label),\n",
        "                                                 id2label=answer2label,\n",
        "                                                 label2id=label2answer)\n",
        "\n",
        "model.to(device)\n",
        "\n",
        "optim_type = \"adamw\"\n",
        "learning_rate = 5e-5\n",
        "weight_decay = 0.01\n",
        "\n",
        "from torch.optim.lr_scheduler import LambdaLR\n",
        "\n",
        "# Define a lambda function for linear learning rate decay\n",
        "# This function will linearly decrease the learning rate from its initial value to 0 over the course of an epoch\n",
        "lr_lambda = lambda epoch_progress: 1 - epoch_progress\n",
        "\n",
        "# Create an optimizer\n",
        "if optim_type == \"adamw\":\n",
        "    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)\n",
        "\n",
        "# Create a scheduler\n",
        "scheduler = LambdaLR(optimizer, lr_lambda)"
      ],
      "metadata": {
        "id": "pnjI563PMMST"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "save_dir = \"/content/drive/MyDrive/saved_models_vilt_V2\"\n",
        "if not os.path.exists(save_dir):\n",
        "    os.makedirs(save_dir)\n",
        "\n",
        "for epoch in range(5):\n",
        "    avg_loss = train(model, train_loader, optimizer, scheduler)\n",
        "    print(f\"Epoch: {epoch}, Loss: {avg_loss:.4f}\")\n",
        "\n",
        "    # Save the model weights with the current epoch number\n",
        "    save_path = os.path.join(save_dir, f\"model_epoch_{epoch}.pth\")\n",
        "    torch.save(model.state_dict(), save_path)\n",
        "\n",
        "print(\"Training finished.\")"
      ],
      "metadata": {
        "id": "JjJCpxxmMZy7"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Dataset & DataLoader\n",
        "test_dataset = VQADataset(test_df, processor, test_img_path, is_test=True)\n",
        "test_loader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=32, shuffle=False)\n",
        "\n",
        "# inference\n",
        "preds = inference(model, test_loader)\n",
        "\n",
        "sample_submission['answer'] = preds\n",
        "sample_submission.to_csv('/content/drive/MyDrive/saved_models_vilt_V2/submission.csv', index=False)"
      ],
      "metadata": {
        "id": "egaNQBifeGBd"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}