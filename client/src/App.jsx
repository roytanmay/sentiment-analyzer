import { useState } from "react";
import axios from "axios";
import styles from "./App.module.css";
import Modal from "./components/modal/Modal";
import BeatLoader from "react-spinners/BeatLoader";

const App = () => {
  const [review, setReview] = useState("");
  // const [sentiment, setSentiment] = useState("");
  const [prediction, setPrediction] = useState(0);
  // const [probability, setProbability] = useState("");
  const [openModal, setOpenModal] = useState(false);
  const [loading, setLoading] = useState(false);

  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL;

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!review.trim()) {
      alert("Enter a review");
      return;
    }

    setLoading(true);

    try {
      const response = await axios.post(`${BACKEND_URL}/predict`, { review });
      // console.log(response.data);
      const data = response.data;
      // setSentiment(data.sentiment);
      setPrediction(data.prediction);
      // setProbability(data.probability);

      setOpenModal(true);
    } catch (error) {
      console.error("Error checking sentiment!!! Try Again Later...");
    }

    setLoading(false);
  };

  return (
    <>
      <div className={styles.container}>
        <h1 className={styles.heading}>Sentiment Analyzer</h1>
        <p className={styles.tagline}>
          Decipher the Sentiment: Positive or Negative
        </p>

        <form onSubmit={handleSubmit} className={styles.form}>
          <textarea
            className={styles.input}
            rows={10}
            cols={30}
            name="review"
            placeholder="Enter review..."
            value={review}
            onChange={(e) => setReview(e.target.value)}
          />

          <div className={styles.btnDiv}>
            <p className={styles.para} style={{ color: "#EE4E4E" }}>
              It may take some time while the model makes the prediction
            </p>

            <button type="submit" className={styles.btn}>
              {loading ? (
                <BeatLoader size={20} color="#ffffff" />
              ) : (
                "Check Sentiment"
              )}
            </button>
          </div>
        </form>

        <h3 className={styles.disclaimer}>: Disclaimer :</h3>
        <p className={styles.para}>
          This sentiment analysis is powered by a logistic regression model with
          a precision score of around 82%. The model is trained on IMDB movie
          reviews data. While we strive for accuracy, the results may not always
          be perfect. Please interpret the results accordingly.
        </p>
      </div>

      <Modal open={openModal} onClose={() => setOpenModal(false)}>
        <div className={styles.modal}>
          {/* <img
            src={prediction === 1 ? "/positive.png" : "/negative.png"}
            className={styles.emoji}
          /> */}

          {prediction === 1 ? (
            <img src="/positive.png" alt="image" className={styles.emoji} />
          ) : (
            <img src="/negative.png" alt="image" className={styles.emoji} />
          )}
          <p className={styles.tagline} style={{ color: "#2c3333" }}>
            {prediction === 1
              ? "Your review radiates positivity."
              : "Your review seems to be on the darker side."}
          </p>
        </div>
      </Modal>
    </>
  );
};

export default App;
