from data_preprocess.category import mushroom


if __name__ == '__main__':
    root_path = "/home/pgj/PycharmProjects/FungusClassification/original_img"

    agaric = mushroom("agaric", root_path)
    bolete = mushroom("bolete", root_path)
    discomycete = mushroom("discomycete", root_path)
    lichen = mushroom("lichen", root_path)

    category_list = [agaric, bolete, discomycete, lichen]

    for category in category_list:
        print("{}共有{}张图片，平均高度{}，平均宽度{}".format(category.category,category.train_count, category.avg_height, category.avg_width))
