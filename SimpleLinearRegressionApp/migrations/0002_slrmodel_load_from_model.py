# Generated by Django 3.0.6 on 2020-07-23 20:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('SimpleLinearRegressionApp', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='slrmodel',
            name='load_from_model',
            field=models.CharField(choices=[(True, 'Yes'), (False, 'No')], default=False, max_length=3),
        ),
    ]
