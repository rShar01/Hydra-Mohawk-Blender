from deit.models import deit_tiny_distilled_patch16_224



if __name__ == "__main__":
    model = deit_tiny_distilled_patch16_224(pretrained=True)
    print("ok")


