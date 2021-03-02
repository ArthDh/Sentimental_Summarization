import torch

class MergeModel(torch.nn.Module):
    '''
            Todo
    '''
    def __init__(self, summary_model, sentiment_model):
        super(MergeModel, self).__init__()
        self.summary_model = summary_model
        self.sentiment_model = sentiment_model
    
    def forward(self, article_id, article_mask, summary_id, summary_mask):
        self.summary_out = self.summary_model(input_ids=article_id, attention_mask =article_mask, labels=summary_id, decoder_attention_mask=summary_mask, return_dict=True)
       
        soft_idx, idx = torch.max(self.summary_out.logits, dim=-1, keepdims= True)
        
        
        mask = torch.zeros_like(self.summary_out.logits).scatter_(-1, idx, 1.).float().detach() + soft_idx - soft_idx.detach()

        self.article_sentiment = self.sentiment_model(article_id)
        self.summary_sentiment = self.sentiment_model(mask)
        
        return [self.summary_out, self.article_sentiment, self.summary_sentiment]
