from bpetrainer import BPETokenizer, getdata
import fire
from pydantic import BaseModel, Field, ValidationError

class DecodeInput(BaseModel):
    ids: list[int, ...] | tuple[int, ...]

class EncodeInput(BaseModel):
    text: str

class TrainInput(BaseModel):
    vocab_size: int = Field(default=100, gt=0)
    data_path: str | None = None

bpe = BPETokenizer()
def train(vocab_size:int = 100, data_path: str | None = None):
    if data_path is None:
        data_path = getdata()

    try:
        inputs = TrainInput(vocab_size=vocab_size, data_path=data_path)
    except ValidationError as e:
        print(f"Error: {e}")
        exit(1)
    with open(inputs.data_path, "r") as f:
        bpe.train(f.read(), inputs.vocab_size)

def encode(text: str):
    try:
        inputs = EncodeInput(text=text)
    except ValidationError as e:
        print(f"Error: {e}")
        exit(1)
    ids = bpe.encode(inputs.text)
    print(ids)

def decode(ids: list | tuple):
    try:
        inputs = DecodeInput(ids=ids)
    except ValidationError as e:
        print(f"Error: {e}")
        exit(1)
    text = bpe.decode(tuple(inputs.ids))
    print(text)

if __name__ == "__main__":
    fire.Fire()

