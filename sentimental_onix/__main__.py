import sys


def main():
    try:
        cmd = sys.argv[1]
        lang = sys.argv[2]
        if cmd == "download":
            print(f"downloading artifacts for lang: {lang}")
            if lang == "en":
                import sentimental_onix.inference.en

                sentimental_onix.inference.en.download()
            else:
                raise NotImplementedError(f"language not implemented: {lang}")
        else:
            raise NotImplementedError(f"command not implemented: {cmd}")
    except:
        print("usage: python -m sentimental_onix download en")


if __name__ == "__main__":
    main()
