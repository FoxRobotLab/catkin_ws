import tensorflow as tf
import os
import cv2
import numpy as np
import rospy
import time


################## Evaluate ##################
# TODO: Maybe write a evaluation function that uses a batch of images with labels and decide precision
# TODO: every n step --> evaluate https://www.tensorflow.org/versions/r1.1/get_started/monitors

def predict_turtlebot_image(olricnn):
    ### Get checkpoint path
    if (not olricnn.checkpoint_name is None):
        ckpt_path = os.path.join(
            olricnn.checkpoint_dir,
            olricnn.checkpoint_name
        )
    else:
        ckpt_path = tf.train.latest_checkpoint(olricnn.checkpoint_dir)

    ### Create input image placeholder
    image = tf.placeholder(tf.float32, shape=(1, olricnn.image_size, olricnn.image_size, olricnn.image_depth), name="input")

    ### Make the graph to use and initialize
    softmax = olricnn.cnn_headings(image, mode="PREDICT", reuse=None)
    init = tf.global_variables_initializer()
    saver = tf.train.import_meta_graph(olricnn.checkpoint_dir + olricnn.checkpoint_name + ".meta")

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, ckpt_path)

        while (not rospy.is_shutdown()):
            ### Get the image from the robot and clean it
            turtle_image, _ = olricnn.robot.getImage()
            eval_image = olricnn.clean_turtlebot_image(turtle_image)
            ### Feed the image and run the session to get softmax_linear
            pred = sess.run(softmax, feed_dict={image: eval_image})
            # tf.reset_default_graph()

            ### Manually pick the label with higher probability (argmax) and
            ### put on the image
            print(pred[0])
            if pred[0][olricnn.label_dict["carpet"]] > pred[0][olricnn.label_dict["tile"]]:
                pred_str = "carpet"
            else:
                pred_str = "tile"

            text_size = cv2.getTextSize(pred_str, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = int((turtle_image.shape[1] - text_size[0]) / 2)
            text_y = int((turtle_image.shape[0] + text_size[1]) / 2)

            cv2.putText(
                img=turtle_image,
                text=pred_str,
                org=(text_x, text_y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 255, 0),
                thickness=2)

            ### Show Turtlebot image with the prediction string
            cv2.imshow("Test Image", turtle_image)
            key = cv2.waitKey(10)
            ch = chr(key & 0xFF)
            if (ch == "q"):
                break

            ### Too many images get fetched without this...
            time.sleep(0.5)
        """
        print("*********** TIME REPORT ***************")
        print("tensor_eval", str(tensor_eval_t2-tensor_eval_t1))
        print("init_graph", str(init_graph_t2 - init_graph_t1))
        print("restore_ckpt", str(restore_ckpt_t2 - restore_ckpt_t1))
        print("robot_img", str(robot_img_t2 - robot_img_t1))

        tensor_eval     0.0883851051331
        init_graph      0.157771110535
        restore_ckpt    0.2871530056
        robot_img       0.00103998184204
        """
        cv2.destroyAllWindows()
        olricnn.robot.stop()


def clean_turtlebot_image(self, image):
    """
    Ensure that the image has right shape and datatype (tf.float32) to prepare it to be fed to inference()
    :param image: nparray with 3 channels
    :return: image (nparray) with shape of (self.image_size, self.image_size, self.image_depth)
    """
    resized_image = cv2.resize(image, (self.image_size, self.image_size))
    cleaned_image = np.array([resized_image], dtype="float").reshape(1, self.image_size, self.image_size,
                                                                     self.image_depth)
    return cleaned_image
