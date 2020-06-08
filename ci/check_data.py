def data_reader(part_start=0, part_end=10, is_test=False):
    data_names = listdir(preprocess_path)
    data_part = data_names[len(data_names) * part_start // 10 : len(data_names) * part_end // 10]
    random.shuffle(data_part)

    def reader():
        for data_name in tqdm(data_part):
            data = np.load(os.path.join(preprocess_path, data_name))
            vol = data[0:3, :, :].reshape(3, 512, 512).astype("float32")
            lab = data[3, :, :].reshape(1, 512, 512).astype("int32")
            if args.windowlize:
                vol = windowlize_image(vol, 200, 70)  # 肝脏常用
            yield vol, lab

    return reader
