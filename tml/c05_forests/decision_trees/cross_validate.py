from classifier import MushroomTree, MushroomForest, MushroomRegression


def main():
    data = './data/agaricus-lepiota.data'
    folds = 5

    print("Calculating score for decision tree")

    tree = MushroomTree(data)
    print(tree.validate(folds))

    print("Calculating score for random forest method")

    forest = MushroomForest(data)
    print(forest.validate(folds))

    print("Calculating score for regression tree")

    regression = MushroomRegression(data)
    print(regression.validate(folds))


if __name__ == "__main__":
    main()
