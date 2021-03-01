import torch

class MergeModel(torch.nn.Module):
    '''
            Todo
    '''
    def __init__(self, summary_model, sentiment_model, freeze_sentiment=False):
        super(MergeModel, self).__init__()
        self.summary_model = summary_model
        self.sentiment_model = sentiment_model
        self.freeze_sentiment = freeze_sentiment
        if self.freeze_sentiment:
            for param in self.sentiment_model.parameters():
                param.requires_grad = False
    
    def forward(self, article_id, article_mask, summary_id, summary_mask):
        self.summary_out = self.summary_model(article_id,  article_mask, summary_id, summary_mask)
       
        idx = torch.argmax(self.summary_out.logits, dim=-1, keepdims= True)
        
        mask = torch.zeros_like(self.summary_out.logits).scatter_(-1, idx, 1.).float().detach() + self.summary_out.logits - self.summary_out.logits.detach()

        self.article_sentiment = self.sentiment_model(article_id)
        self.summary_sentiment = self.sentiment_model(mask)
        
        return [self.summary_out, self.article_sentiment, self.summary_sentiment]
