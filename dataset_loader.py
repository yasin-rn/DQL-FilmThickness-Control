import pandas as pd
import json
import numpy as np

class DatasetLoader:
    def __init__(self, filename: str):
        with open(filename, "r") as f:
            data = json.load(f)

        self.data_frame = pd.DataFrame({
            "ActuatorPositions": data["ActuatorPositions"],
            "ActuatorDeviations": data["ActuatorDeviations"],
            "ActuatorActions": data["ActuatorActions"],
            "ThiknessProfiles": data["ThiknessProfiles"],
            "Averages": data["Averages"]
        })

        self.data_length = len(self.data_frame)
        print(f"DataFrame {self.data_length} satır ile başlatıldı.")

    def _generate_1d_sinusoidal_pe(self, length: int, internal_pe_dim: int = 16) -> np.ndarray:
        """
        Verilen bir uzunluk için 1D sinüzoidal pozisyonel encoding vektörü üretir.
        Bu vektör, orijinal 1D veri vektörüne eklenebilir.
        """
        if length == 0:
            return np.array([], dtype=np.float32)
        if internal_pe_dim <= 0 : # Ensure internal_pe_dim is positive
            internal_pe_dim = 2 # Default to a small even number if invalid

        # internal_pe_dim'in çift sayı olmasını garantile (sin/cos çiftleri için)
        if internal_pe_dim % 2 != 0:
            internal_pe_dim +=1
            
        position = np.arange(length, dtype=np.float32).reshape(-1, 1) # Shape: (length, 1)
        div_term = np.exp(np.arange(0, internal_pe_dim, 2, dtype=np.float32) * \
                          -(np.log(10000.0) / internal_pe_dim)) # Shape: (internal_pe_dim/2,)

        pe_matrix = np.zeros((length, internal_pe_dim), dtype=np.float32)
        pe_matrix[:, 0::2] = np.sin(position * div_term)
        pe_matrix[:, 1::2] = np.cos(position * div_term)

        # (length, internal_pe_dim) matrisini (length,) vektörüne indirge (örn: satırların toplamı)
        # Bu, her pozisyon için tek bir PE değeri sağlar.
        # Farklı indirgeme yöntemleri (örn: ortalama) veya PE'nin farklı bileşenlerini kullanmak da mümkündür.
        pe_vector = np.sum(pe_matrix, axis=1)
        
        # Orijinal veriyle toplanabilir hale getirmek için normalize etmeyi veya ölçeklendirmeyi düşünebilirsiniz.
        # Örneğin: (pe_vector - np.mean(pe_vector)) / np.std(pe_vector) * 0.1
        # Bu adım, verinizin ve PE'nin göreceli büyüklüklerine bağlıdır.
        # Şimdilik ham PE vektörünü döndürelim.
        return pe_vector

    def positional_encoding(self, data: np.ndarray, time_step_seq_idx: int) -> np.ndarray:
        """
        Girdi verisine özellik-içi (intra-feature) pozisyonel encoding ekler.
        'time_step_seq_idx' şu anda bu implementasyonda kullanılmıyor,
        çünkü özellik-içi PE genellikle özelliğin kendi içindeki pozisyonlara dayanır.
        """
        feature_length = len(data)
        if feature_length == 0:
            return data
        
        # Her bir özellik vektörünün kendi içindeki sıralı/uzamsal bilgiyi korumak için PE
        # Örneğin, actuator_positions (48 eleman) için kendi içinde bir PE,
        # thickness_profile (360 eleman) için kendi içinde ayrı bir PE.
        pe_vector = self._generate_1d_sinusoidal_pe(feature_length)
        
        return data + pe_vector

    def get_seq_data(self, seq_len: int, input_headers: list, output_headers: list, apply_input_positional_encoding_header: list):
        # Proje planında durum tensörü için sequence_length = 5 belirtilmişti.
        # Kodunuzda esnek olması iyidir, ancak eğitim sırasında bu değere dikkat edin.
        # Örnek çağrıda seq_len=8 kullanılmış.

        # index = seq_len # Bu satır aslında kullanılmıyor gibi duruyor.
        # Kaynak DataFrame'den veri çekerken kullanılacak olan asıl başlangıç indeksi
        # (row + seq_len - 1) - (seq -1) = row + seq_len - seq olacaktır.
        # Ya da daha basitçe, her bir sekans [row, row+1, ..., row+seq_len-1] zaman adımlarını içerir.
        # Döngünüz range(self.data_length-seq_len) olduğu için, son tam sekans
        # (self.data_length-seq_len-1) indisinden başlar ve
        # (self.data_length-seq_len-1) + seq_len -1 = self.data_length - 2 'ye kadar gider.
        # Bu, DataFrame'in son satırını kullanmaz. Eğer son satırı da dahil etmek isterseniz
        # range(self.data_length - seq_len + 1) ve erişimlerde dikkatli olmalısınız.
        # Mevcut haliyle bir sorun teşkil etmiyor, sadece bir not.

        all_input_sequences = []
        all_output_sequences = []

        # (self.data_length - seq_len) toplamda bu kadar sekans üretebiliriz.
        for start_row_idx in range(self.data_length - seq_len +1): # son olası başlangıç noktasına kadar git
            
            current_input_sequence = [] # Bu, [seq_len, 456] tensörüne karşılık gelecek
            current_output_sequence = [] # Bu, [seq_len, 48, 3] tensörüne karşılık gelecek (davranış klonlama için)

            for i in range(seq_len):
                current_time_step = start_row_idx + i # Sekans içindeki her bir zaman adımının DataFrame'deki satır indeksi
                
                concatenated_input_features_for_step = []
                for header_idx, input_header in enumerate(input_headers):
                    feature_data = np.array(self.data_frame.loc[current_time_step, input_header], dtype=np.float32)
                    if apply_input_positional_encoding_header[header_idx]:
                        # 'i' burada sekans içindeki göreceli zaman adımını (0'dan seq_len-1'e) temsil eder.
                        # positional_encoding fonksiyonumuz bunu şu an kullanmıyor, özellik içi PE yapıyor.
                        feature_data = self.positional_encoding(feature_data, i) 
                    concatenated_input_features_for_step.extend(feature_data) # .extend() listeler için, NumPy arrayleri için np.concatenate
                
                # Yukarıdaki extend yerine doğrudan np.concatenate kullanalım:
                step_input_row_list = []
                for header_idx, input_header in enumerate(input_headers):
                    feature_data = np.array(self.data_frame.loc[current_time_step, input_header], dtype=np.float32)
                    if apply_input_positional_encoding_header[header_idx]:
                        feature_data = self.positional_encoding(feature_data, i) # i: sekans içindeki zaman indeksi
                    step_input_row_list.append(feature_data)
                
                # Tüm özellikleri birleştirerek 456 elemanlı durumu oluştur
                final_input_row_for_step = np.concatenate(step_input_row_list)
                current_input_sequence.append(final_input_row_for_step)

                # Çıktı (Aksiyon) verisini işle
                # Davranış klonlama için her bir zaman adımındaki aksiyonu alıyoruz.
                # Genellikle model, bir sekans durumuna karşılık bir sonraki aksiyonu veya bir dizi aksiyonu tahmin eder.
                # Eğer her adımdaki aksiyonu hedefliyorsak:
                step_output_row_list = []
                for output_header in output_headers: # output_headers = ["ActuatorActions"]
                    action_data = np.array(self.data_frame.loc[current_time_step, output_header], dtype=int)
                    # one_hot_encode (48,) şeklindeki aksiyonları (48,3) şekline getirir
                    one_hot_actions = self.one_hot_encode(action_data, num_classes=3)
                    step_output_row_list.append(one_hot_actions) 
                
                # Eğer birden fazla output_header varsa birleştirme gerekebilir, şu an tek header var.
                final_output_row_for_step = step_output_row_list[0] # (48,3) şeklinde
                current_output_sequence.append(final_output_row_for_step)

            all_input_sequences.append(current_input_sequence)
            all_output_sequences.append(current_output_sequence)
            
        return np.array(all_input_sequences, dtype=np.float32), np.array(all_output_sequences, dtype=np.float32)

    def one_hot_encode(self, actions: np.ndarray, num_classes: int = 3) -> np.ndarray:
        # 'actions' (48,) şeklinde bir NumPy array, her eleman bir class index (0, 1, veya 2).
        actions = actions.astype(int)
        # np.eye(num_classes) -> (3,3) birim matris
        # np.eye(num_classes)[actions] -> (48,3) şeklinde one-hot matrix oluşturur.
        one_hot_matrix = np.eye(num_classes, dtype=np.float32)[actions]
        return one_hot_matrix # Çıktı şekli (48, 3) olacak
    

