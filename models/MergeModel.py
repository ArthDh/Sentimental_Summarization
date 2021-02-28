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
        self.summary_out = self.summary_model(article_id,  article_mask, summary_id, summary_mask)
       
        prob_logits,_ = torch.max(self.summary_out.logits,-1)

        self.article_sentiment = self.sentiment_model(self.summary_out.logits)
        self.summary_sentiment = self.sentiment_model(self.summary_out.logits)
        
        return [self.summary_out, self.article_sentiment, self.summary_sentiment]