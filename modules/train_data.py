def train_from_buffer(model, turns, buff, class_store,
                      batch_size=32, epochs=5):

    import numpy as np
    from sklearn.model_selection import train_test_split
    for epoch in range(epochs):
        print(epoch)
        batch_losses = []
        batch_accs = []
        batch_Top5 =[]
        for _ in range(int(turns)):
            # ---- sampling ----
            buff.sample_data(batch_size)
            buff.reservoir_sampling(batch_size)
            batch_data = buff.reservoir   # list of (idx, Series)
            # ---- extract raw labels ----
            labels_in_batch = np.array(
                 [str(sample[1].iloc[68]) for sample in batch_data],
                    dtype=object
                         )
            print("HEllo",labels_in_batch)
            # ---- detect new classes ----
            new_labels = set(labels_in_batch) - class_store
            label_to_index = {label: idx for idx, label in enumerate(class_store)}
            print("New Labels",new_labels)
            if new_labels:
                print("🚨 New classes detected:", new_labels)
                # save old weights
                old_weights = model.get_weights()
                # update class store
                class_store.update(new_labels)
                label_to_index = {
                    label: idx for idx, label in enumerate(sorted(class_store))
                      }
                # rebuild model with larger output layer
                new_model = create_model(num_class=len(class_store))
                # copy all layers except output layer
                new_weights = new_model.get_weights()
                new_weights[:-2] = old_weights[:-2]
                new_model.set_weights(new_weights)
                model = new_model

            # ---- features ----
            X = np.vstack([
               sample[1].iloc[0:68].astype(float).values
              for sample in batch_data
               ]).astype(np.float32)

            # ---- integer labels ----
            y = np.array(
               [label_to_index[str(sample[1].iloc[68])] for sample in batch_data],
               dtype=np.int32
                             )
            print(X,y)
            # ---- split ----
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # ---- train ----
            loss, acc,top5= model.train_on_batch(X_train, y_train)
            print(loss,acc)
            batch_losses.append(loss)
            batch_accs.append(acc)
            batch_Top5.append(top5)

        # ---- epoch summary ----
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {np.mean(batch_losses):.4f} | "
            f"Acc: {np.mean(batch_accs):.4f}"
            f"Top-5 Acc: {top5:.4f}"
        )

        # ---- validation ----
        val_loss, val_acc,val_top5 = model.evaluate(X_val, y_val, verbose=0)
        print(f"Validation | Loss: {val_loss:.4f} | Acc: {val_acc:.4f}|Top-5:{val_top5:.4f}")

    return model
