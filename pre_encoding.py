import torch.nn as nn


class Pre_encoding(nn.Module):
    def __init__(
            self, prot_encoder, drug_encoder, args
    ):
        """Constructor for the model.

        Args:
            prot_encoder (_type_): Protein sturcture-aware sequence encoder.
            drug_encoder (_type_): Drug SFLFIES encoder.
            args (_type_): _description_
        """
        super(Pre_encoding, self).__init__()
        self.prot_encoder = prot_encoder
        self.drug_encoder = drug_encoder
        
    def encoding(self, prot_input_ids, prot_attention_mask, drug_input_ids, drug_attention_mask):
        # Process protein encoder with hidden state output
        prot_embed = self.prot_encoder(
            input_ids=prot_input_ids, 
            attention_mask=prot_attention_mask, 
            output_hidden_states=True,  # Request hidden states
            return_dict=True
        ).hidden_states[-1]

        # prot_embed = self.prot_encoder(
        #     input_ids=prot_input_ids, attention_mask=prot_attention_mask, return_dict=True
        # ).logits
        # prot_embed = self.prot_reg(prot_embed)

        drug_embed = self.drug_encoder(
            input_ids=drug_input_ids, attention_mask=drug_attention_mask, return_dict=True
        ).last_hidden_state
        # drug_embed = self.drug_encoder.encode(df['SELFIES'], return_torch=True)
        
        return prot_embed, drug_embed